import argparse

from train import Trainer
from eval import test_unet
from load_data import dataload

def parse():
    parser = argparse.ArgumentParser(description='LeNet')
    
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default='./dataset')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()

    return args

def main():
    args = parse()
    print(args)

    train_data, val_data = dataload(args.batch_size)

    if args.mode == 'train':
        trainer = Trainer(train_data, 
                        args.num_epochs, 
                        args.lr, 
                        args.gpu_id, 
                        args.save_model, 
                        args.eval, 
                        args.tensorboard)
        
        trainer.train_unet()
        
        if args.tensorboard:
            trainer.launch_tensorboard()
    
    elif args.mode == 'val':
        assert args.model_path is not None, 'Insert model path. e.g. --model_path ./model.pth'
        test_unet(val_data, args.model_path)

if __name__ == '__main__':
    main()
