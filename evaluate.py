from pathlib import Path

import argparse
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import StandardScaler

from common_types import TAMHyperParameters, TrainingSettings
from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator
from gnn import GNN_TAM


def parse_args():
    parser = argparse.ArgumentParser(description='model_inference')
    parser.add_argument('--checkpoint_id', type=str)
    # parser.add_argument('--dataset', type=str, default='reinartz_tep')
    # parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    # parser.add_argument('--name', type=str, default='gnn1')
    return parser.parse_args()


def inference():
    args = parse_args()

    # Getting model hyper_params
    instance_dir = Path('saved_models') / args.checkpoint_id.split('_')[0]
    checkpoint_dir = Path('saved_models', *args.checkpoint_id.split('_'))
    with open(instance_dir/'hyperparams.pkl', mode='rb') as hyperparams_file:
        hyper_params: TAMHyperParameters = pickle.load(hyperparams_file)
    with open(instance_dir/'training_settings.pkl', mode='rb') as training_settings_file:
        training_settings: TrainingSettings = pickle.load(training_settings_file)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Data preparation:
    dataset = FDDDataset(name=training_settings.dataset)
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)
    test_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.test_mask,
        window_size=hyper_params.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=True
    )
    # Load saved model:
    state = torch.load(checkpoint_dir/'state.pt')
    model = GNN_TAM(hyper_params=hyper_params)
    model.load_state_dict(state['model_state'])
    model.to(device)

    # Inference:
    model.eval()
    preds = []
    test_labels = []
    for test_ts, test_index, test_label in test_dl:
        ts = torch.FloatTensor(test_ts).to(device)
        ts = torch.transpose(ts, 1, 2)
        with torch.no_grad():
            logits = model(ts)
        pred = logits.argmax(axis=1).cpu().numpy()
        preds.append(pd.Series(pred, index=test_index))
        test_labels.append(pd.Series(test_label, index=test_index))
    pred = pd.concat(preds)
    test_label = pd.concat(test_labels)
    
    evaluator = FDDEvaluator(step_size=1)
    evaluator.print_metrics(test_label, pred)


if __name__ == '__main__':
    inference()
