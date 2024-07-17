import argparse
import subprocess

from train import Trainer
from load_data import data_MNIST

def parse():
    parser = argparse.ArgumentParser(description='LeNet')

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default='./dataset')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--tensorboard', type=bool, default=True)

    args = parser.parse_args()

    return args

def main():
    args = parse()
    print(args)

    train_data, test_data = data_MNIST(args.batch_size)

    trainer = Trainer(train_data, 
                      test_data, 
                      args.num_epochs, 
                      args.lr, 
                      args.gpu_id, 
                      args.save_model, 
                      args.eval, 
                      args.tensorboard)
    
    trainer.train_lenet()
    
    if args.tensorboard:
        trainer.launch_tensorboard()
    
if __name__ == '__main__':
    main()
