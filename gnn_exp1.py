import os
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

from gnn1 import run_experiment
from config import args  

# Define a single directory for storing experiment results
RESULTS_DIR = 'C:\\Users\\vuppu\\malnet-graph\\results'


def model_search(gpu, malnet_tiny, group, metric, epochs, model, K, num_layers, hidden_dim, lr, dropout, train_ratio):
    short_params = {
        'node_feature': 'nf',
        'directed_graph': 'dg',
        'remove_isolates': 'ri',
        'lcc_only': 'lcc',
        'add_self_loops': 'asl',
    }
    # Update args with function arguments
    args.update({
        'gpu': gpu,
        'batch_size': 64,

        'node_feature': 'ldp', 
        'directed_graph': True,
        'remove_isolates': False,
        'lcc_only': False,
        'add_self_loops': True,

        'model': model,
        'K': K,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,

        'metric': metric,
        'lr': lr,
        'dropout': dropout,
        'epochs': epochs,

        'group': group,
        'train_ratio': train_ratio,
        'malnet_tiny': malnet_tiny
    })

    # Set CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])

    # Replace long parameter names with abbreviations in directory paths
    dir_args = {short_params.get(k, k): v for k, v in args.items()}
    log_dir = os.path.join(RESULTS_DIR, "_".join([f"{key}={value}" for key, value in dir_args.items()]))
    args['log_dir'] = log_dir

    # Run experiment and return results
    val_score, test_score, param_count, run_time = run_experiment(args)
    return args, val_score, test_score, param_count, run_time


def preprocess_search(gpu, epochs, node_feature, directed_graph, remove_isolates, lcc_only, add_self_loops, model='gcn', K=0, hidden_dim=32, num_layers=3, lr=0.0001, dropout=0):
    args.update({
        'gpu': gpu,
        'batch_size': 128,

        'node_feature': node_feature,
        'directed_graph': directed_graph,
        'remove_isolates': remove_isolates,
        'lcc_only': lcc_only,
        'add_self_loops': add_self_loops,

        'model': model,
        'K': K,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,

        'lr': lr,
        'dropout': dropout,
        'epochs': epochs,

        'group': 'type',
        'train_ratio': 1.0,
        'malnet_tiny': True
    })


def search_all_preprocess():
    epochs = 250
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]

    # Define tqdm progress bars for parallel function calls
    progress_bars = [tqdm(total=len(gpus)) for _ in range(5)]

    # Test node features
    feature_list = ['ldp', 'constant', 'degree']
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature=feature, directed_graph=True, remove_isolates=True, lcc_only=False, add_self_loops=False)
        for idx, feature in enumerate(progress_bars[0]))

    # Test directed graph
    directed_values = [True, False]
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=directed, remove_isolates=True, lcc_only=False, add_self_loops=False)
        for idx, directed in enumerate(progress_bars[1]))

    # Test isolates
    isolates_values = [True, False]
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=True, remove_isolates=isolates, lcc_only=False, add_self_loops=False)
        for idx, isolates in enumerate(progress_bars[2]))

    # Test lcc
    lcc_values = [True, False]
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=False, remove_isolates=True, lcc_only=lcc, add_self_loops=False)
        for idx, lcc in enumerate(progress_bars[3]))

    # Test self loops
    self_loops_values = [True, False]
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=True, remove_isolates=True, lcc_only=False, add_self_loops=self_loops)
        for idx, self_loops in enumerate(progress_bars[4]))


def search_all_models():
    gpus = [2]
    models = ['gin']
    layers = [5]
    hidden_dims = [64]
    learning_rates = [0.0001]
    dropouts = [0]
    epochs = 250
    metric = 'macro-F1'
    groups = ['family']
    malnet_tiny = False
    train_ratios = [1.0]

    combinations = list(itertools.product(*[groups, models, layers, hidden_dims, learning_rates, dropouts, train_ratios]))

    results = Parallel(n_jobs=len(combinations))(
        delayed(model_search)(gpus[idx % len(gpus)], malnet_tiny, group, metric, epochs, model=model, K=0, num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, dropout=dropout, train_ratio=train_ratio)
        for idx, (group, model, num_layers, hidden_dim, lr, dropout, train_ratio) in enumerate(tqdm(combinations)))

    for (args, val_score, test_score, param_count, run_time) in results:
        print('Tiny={}, group={}, train_ratio={}, model={}, epochs={}, run time={} seconds, # parameters={}, layers={}, hidden_dims={}, learning_rate={}, dropout={}, val_score={}, test_score={}'.format(
            args['malnet_tiny'], args['group'], args['train_ratio'], args['model'], args['epochs'], run_time, param_count, args['num_layers'], args['hidden_dim'], args['lr'], args['dropout'], val_score, test_score))


def run_best_models():
    epochs = 200
    gpus = [2, 3, 4, 5]
    metric = 'macro-F1'
    group = 'family'
    malnet_tiny = True

    combinations = [['gin', 0, 3, 64, 0.001, 0.5], ['sgc', 1, 3, 64, 0.001, 0.5], ['gcn', 0, 5, 64, 0.001, 0.5], ['mlp', 0, 1, 128, 0.001, 0], ['graphsage', 0, 5, 128, 0.0001, 0]]

    results = Parallel(n_jobs=len(combinations))(
        delayed(model_search)(gpus[idx % len(gpus)], malnet_tiny, group, metric, epochs, model=model, K=K, num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, dropout=dropout)
        for idx, (model, K, num_layers, hidden_dim, lr, dropout) in enumerate(tqdm(combinations)))

    for (args, val_score, test_score, param_count, run_time) in results:
            print('Tiny={}, group={}, train_ratio={}, model={}, epochs={}, run time={} seconds, # parameters={}, layers={}, hidden_dims={}, learning_rate={}, dropout={}, val_score={}, test_score={}'.format(
                args['malnet_tiny'], args['group'], args['train_ratio'], args['model'], args['epochs'], run_time, param_count, args['num_layers'], args['hidden_dim'], args['lr'], args['dropout'], val_score, test_score))


if __name__ == '__main__':
    search_all_preprocess()
    search_all_models()
    run_best_models()
