import pickle
import signal
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import argparse
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm, trange

from common_types import TAMHyperParameters, TrainingSettings, CheckpointInfo
from fddbenchmark import FDDDataset, FDDDataloader
from gnn import GNN_TAM


def parse_args():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--dataset', type=str, default='reinartz_tep')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_gnn', type=int, default=1)
    parser.add_argument('--gsl_type', type=str, default='directed')
    parser.add_argument('--n_hidden', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--k_additional', type=int, default=None)
    parser.add_argument('--name', type=str, default='gnn')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=5)
    parser.add_argument('--tensorboard_enabled', type=bool, default=False)
    return parser.parse_args()

def tensorboard_log():
    pass

def gen_unique_directory_name(parent_dir):
    present_names = list(parent_dir.iterdir())
    cntr = 32
    while (uid := uuid.uuid4().hex[:16]) in present_names:
        cntr -= 1
        if cntr == 0:
            raise RuntimeError('Too much attempts to create unique directory')
    return uid


def save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    checkpoint_info: CheckpointInfo,
    instance_dir
):
     
    checkpoint_dir = instance_dir / gen_unique_directory_name(instance_dir)
    
    assert not checkpoint_dir.exists(), f'Directory for checkpoint already exists: "{checkpoint_dir}"'
    checkpoint_dir.mkdir(parents=False)

    state_path = checkpoint_dir / 'state.pt'
    checkpoint_info_path = checkpoint_dir / 'info.pkl'
    torch.save(
        {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        },
        state_path
    )
    with open(checkpoint_info_path, mode='wb') as checkpoint_info_file:
        pickle.dump(checkpoint_info, checkpoint_info_file)
    

def save_model(
    *,
    model,
    optimizer,
    scheduler,
    checkpoint_info: CheckpointInfo,
    hyper_params: TAMHyperParameters,
    training_settings: TrainingSettings, 
    instance_dir,
):
    if not instance_dir.exists():
        instance_dir.mkdir(parents=False)

    hyperparams_path = instance_dir / 'hyperparams.pkl' 
    if not hyperparams_path.exists():
        with open(hyperparams_path, mode='wb') as hyperparams_file:
            pickle.dump(hyper_params, hyperparams_file)

    training_settings_path = instance_dir / 'training_settings.pkl'
    if not training_settings_path.exists():
        with open(training_settings_path, mode='wb') as training_settings_file:
            pickle.dump(training_settings, training_settings_file)
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_info=checkpoint_info,
        instance_dir=instance_dir,
    )

def get_split_by(dataset: FDDDataset):
    return (
        list(list(range(53))[:41]),
        list(list(range(53))[41:]),
    ) 


def train():
    args = parse_args()

    # Validate that save directory is absent in advance
    save_models_dir = Path('saved_models')
    save_models_dir.mkdir(parents=False, exist_ok=True)
    instance_dir = save_models_dir / gen_unique_directory_name(save_models_dir)
    
    assert not instance_dir.exists(), f'Directory for new instance already exists: "{instance_dir}"'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Data preparation:
    dataset = FDDDataset(name=args.dataset)
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)
    n_nodes = dataset.df.shape[1]
    n_classes = len(set(dataset.label))
    train_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.train_mask,
        window_size=args.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Model creation:
    training_settings = TrainingSettings(
        dataset=args.dataset,
        n_epochs=args.n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        tensorboard_enabled=args.tensorboard_enabled,
    )
    hyper_parameters = TAMHyperParameters(
        n_nodes=n_nodes,
        window_size=args.window_size,
        n_classes=n_classes,
        n_gnn=args.n_gnn,
        gsl_type=args.gsl_type,
        n_hidden=args.n_hidden,
        alpha=args.alpha,
        k=args.k,
        split_by=get_split_by(dataset),
        k_additional=args.k_additional,
    )
    model = GNN_TAM(hyper_params=hyper_parameters)
    model.to(device)

    # Training:
    checkpoint_info = CheckpointInfo()
    start_time = time.time()

    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=1, threshold=0.001)

    def signal_handler(sig, frame):
        if checkpoint_info.epochs_elapsed > 0:
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_info=checkpoint_info,
                hyper_params=hyper_parameters,
                training_settings=training_settings,
                instance_dir=instance_dir,
            )
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    weight = torch.ones(n_classes) * 0.5
    weight[1:] /= (n_classes - 1)
    for e in trange(args.n_epochs, desc="Epochs ..."):
        av_loss = []
        for train_ts, train_index, train_label in tqdm(train_dl):
            ts = torch.FloatTensor(train_ts).to(device)
            ts = torch.transpose(ts, 1, 2)
            train_label = torch.LongTensor(train_label).to(device)
            logits = model(ts)
            loss = F.cross_entropy(logits, train_label, weight=weight.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            av_loss.append(loss.item())
        train_avg_loss = sum(av_loss)/len(av_loss)
        checkpoint_info.epochs_elapsed += 1
        checkpoint_info.total_seconds_elapsed = time.time() - start_time
        checkpoint_info.train_loss = train_avg_loss
        checkpoint_info.train_loss_history.append(train_avg_loss)
        scheduler.step(train_avg_loss) # TODO: Change to validation loss
        if (e + 1) % training_settings.checkpoint_every_n_epochs == 0 and (e+1) != args.n_epochs:
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_info=checkpoint_info,
                hyper_params=hyper_parameters,
                training_settings=training_settings,
                instance_dir=instance_dir,
            )
        print(f'Epoch: {e+1:2d}/{args.n_epochs}, average CE loss: {train_avg_loss:.4f}')


    save_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_info=checkpoint_info,
        hyper_params=hyper_parameters,
        training_settings=training_settings,
        instance_dir=instance_dir,
    )


if __name__ == '__main__':
    train()
