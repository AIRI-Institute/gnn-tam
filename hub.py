import time
from pathlib import Path

import click
import pickle
import torch
from prettytable import PrettyTable

from common_types import TAMHyperParameters, TrainingSettings, CheckpointInfo
from gnn import GNN_TAM


@click.group()
def cli():
    pass

@cli.group()
def ls():
    pass

@ls.command()
def instances():
    pass

@ls.command()
def checkpoints():
    saved_models_dir = Path('saved_models')
    assert saved_models_dir.exists()

    table = PrettyTable()
    table.field_names = ['Id', 'Train loss', 'Validation loss', 'N epochs', 'Time per epoch']
    for instance_dir in saved_models_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        with open(instance_dir/'hyperparams.pkl', 'rb') as hyperparams_file:
            hyper_params: TAMHyperParameters = pickle.load(hyperparams_file)
        with open(instance_dir/'training_settings.pkl', 'rb') as training_settings_file:
            training_settings: TrainingSettings = pickle.load(training_settings_file)
        for checkpoint_dir in instance_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            with open(checkpoint_dir/'info.pkl', 'rb') as checkpoint_info_file:
                checkpoint_info: CheckpointInfo = pickle.load(checkpoint_info_file)
            if checkpoint_info.epochs_elapsed == 0:
                continue
            table.add_row([
                '_'.join(checkpoint_dir.parts[-2:]),
                checkpoint_info.train_loss,
                checkpoint_info.validation_loss,
                checkpoint_info.epochs_elapsed,
                time.strftime('%H:%M:%S', time.gmtime(checkpoint_info.total_seconds_elapsed//checkpoint_info.epochs_elapsed))
            ])
    print(table)

@cli.command()
@click.option('--checkpoint_id')
def print_adj(checkpoint_id):
    instance_dir = Path('saved_models') / checkpoint_id.split('_')[0]
    checkpoint_dir = Path('saved_models', *checkpoint_id.split('_'))
    with open(instance_dir/'hyperparams.pkl', mode='rb') as hyperparams_file:
        hyper_params: TAMHyperParameters = pickle.load(hyperparams_file)
    with open(instance_dir/'training_settings.pkl', mode='rb') as training_settings_file:
        training_settings: TrainingSettings = pickle.load(training_settings_file)
    
    # Load saved model:
    state = torch.load(checkpoint_dir/'state.pt')
    model = GNN_TAM(hyper_params=hyper_params)
    model.load_state_dict(state['model_state'])
    with torch.no_grad():
        print(model.gsl[0](model.idx)*model.z)


if __name__ == '__main__':
    cli()