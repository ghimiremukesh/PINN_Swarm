# Enable import from parent package
import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')

p.add_argument('--num_epochs', type=int, default=110000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='relu', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=2.5, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=32, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=10000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculum training')
p.add_argument('--counter_end', type=int, default=100000, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=10000, required=False, help='Number of source samples at each time step')


p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=True, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--alpha', default=0.5, help="Temperature parameter for softmax/boltzmann")

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets


if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs


dataset = dataio.SwarmHJI(numpoints=65000, alpha=opt.alpha, u_max=1, u_min=0, t_min=opt.tMin, t_max=opt.tMax,
                          counter_start=opt.counter_start, counter_end=opt.counter_end, pretrain=opt.pretrain,
                          pretrain_iters=opt.pretrain_iters, num_src_samples=opt.num_src_samples, seed=opt.seed)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=9, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)

model.to(device)

loss_fn = loss_functions.initialize_swarm_hji(dataset)

root_path = os.path.join(opt.logging_root, 'swarm_hji/')

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary,epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path,
               loss_fn=loss_fn, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs, validation_fn=None,
               start_epoch=opt.checkpoint_toload)
