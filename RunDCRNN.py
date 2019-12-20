
from Model.DCRNN1.Trainer import Trainer
from Model.DCRNN1.DCRNNModel import DCRNNModel
import argparse
import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.utils.data
import pickle
from RootPATH import base_dir

args = argparse.ArgumentParser(description='PyTorch DCRNN')
args.add_argument('--device', default='cuda:5', type=str, help='indices of GPUs')
args.add_argument('--debug', default=False, type=bool)
#Model details
args.add_argument('--enc_input_dim', default=2, type=int)
args.add_argument('--input_dim', default=2, type=int)
args.add_argument('--dec_input_dim', default=1, type=int)
args.add_argument('--output_dim', default=1, type=int)
args.add_argument('--diffusion_step', default=2, type=int)
args.add_argument('--num_nodes', default=207, type=int)
args.add_argument('--num_rnn_layers', default=2, type=int)
args.add_argument('--rnn_units', default=64, type=int)
args.add_argument('--seq_len', default=12, type=int)
#training
args.add_argument('--batch_size', default=64, type=int)
args.add_argument('--epochs', default=100, type=int)
args.add_argument('--lr_init', default=0.01, type=float)
args.add_argument('--lr_type', default='MultiStepLR', type=str)
args.add_argument('--lr_milestones', default='20,30,40,50', type=str)
args.add_argument('--lr_decay_rate', default=0.1, type=float)
args.add_argument('--max_grad_norm', default=5, type=int)
args.add_argument('--early_stop', default=10, type=int)
args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--plot', default=True, type=bool)
args = args.parse_args()

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_graph():
    graph_pkl_filename = os.path.join(base_dir, 'data/MetrLA/adj_mx_la.pkl')
    sensor_ids, sensor_id_to_ind, adj_mat = load_pickle(graph_pkl_filename)
    return adj_mat

def data_loader(X, Y, batch_size, shuffle=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def load_data(batch_size):
    data_dir = os.path.join(base_dir, 'data/MetrLA/processed')
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    print('Train', data['x_train'].shape, data['y_train'].shape)
    print('Val', data['x_val'].shape, data['y_val'].shape)
    print('Test', data['x_test'].shape, data['y_test'].shape)
    data['train_loader'] = data_loader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = data_loader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    data['test_loader'] = data_loader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    #from Model.DCRNN2.lib.utils import DataLoader
    #data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    #data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    #data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    data['scaler'] = scaler
    return data

def masked_mae_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.ne(labels, null_val)
    mask = mask.to(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_torch(preds=preds, labels=labels, null_val=null_val)
        return mae
    return loss

def main(logger):
    adj_mat = load_graph()
    from lib.graph_laplacian import calculate_scaled_laplacian
    supports = [calculate_scaled_laplacian(adj_mat, lambda_max=None)]
    supports = [torch.tensor(i).to(args.device) for i in supports]
    data = load_data(args.batch_size)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    scaler = data['scaler']
    model = DCRNNModel(supports, num_node=args.num_nodes, input_dim=args.input_dim,
                       hidden_dim=args.rnn_units, out_dim=args.output_dim,
                       order=args.diffusion_step, num_layers=args.num_rnn_layers)
    '''
    from Model.DCRNN2.model.dcrnn_model import DCRNNModel
    model = DCRNNModel(adj_mat, args.batch_size, args.enc_input_dim, args.dec_input_dim,
                       args.diffusion_step, args.num_nodes,args.num_rnn_layers,
                       args.rnn_units, args.seq_len, args.output_dim)
    '''
    model = model.to(args.device)
    from lib.TrainInits import print_model_parameters
    print_model_parameters(model)
    loss = masked_mae_loss(scaler, null_val=0.0)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-3,
                                 weight_decay=0, amsgrad=True)
    lr_milestones = [int(i) for i in list(args.lr_milestones.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_milestones,
                                                        gamma=args.lr_decay_rate)
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, args, logger,
                      lr_scheduler=lr_scheduler)
    trainer.train()
    trainer.test(model, test_loader, scaler, logger, path=trainer.best_path)

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    from lib.logger import get_logger
    #set log dir
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir,'output', current_time)
    if os.path.isdir(log_dir) == False:
        os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    #save args to file
    args_dir = os.path.join(log_dir, 'args.txt')
    with open(args_dir, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logger = get_logger(log_dir, name='DCRNN', debug=args.debug)
    main(logger)






