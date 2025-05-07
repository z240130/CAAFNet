import argparse

import os
import random
import numpy as np
import torch

from networks.CAAFNet import CAAFNet

import warnings

from trainer_POLY import trainer_POLY

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", default="F:/yuyifenge/Pytorch-UNet-master/data/ACDC/")
parser.add_argument('--dataset', type=str,
                    default='POLY', help='experiment_name')
parser.add_argument("--list_dir", default="F:/yuyifenge/Pytorch-UNet-master/data/ACDC/lists_ACDC")
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--output_dir', type=str,
                    default='./model_out',help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=352, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'ACDC': {
            'root_path': args.root_path,
            'list_dir': 'F:\yuyifenge\Pytorch-UNet-master\data\ACDC\lists_ACDC',
            'num_classes': 4,},
        'POLY':{
            'root_path': args.root_path,
            'list_dir': 'F:\yuyifenge\Pytorch-UNet-master\data\ACDC\lists_ACDC',
            'num_classes': 1,
        }

    }

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = CAAFNet(num_classes=args.num_classes).cpu()

    trainer = {'POLY': trainer_POLY}
    trainer[dataset_name](args, net, args.output_dir)  # 传入 net 和输出路径