import argparse
import edege_classifier_model as ecm

def get_args():

    parser = argparse.ArgumentParser(description='Edge classifier')

    # model parameters
    

    # data parameters
    parser.add_argument('--nodes_transform', type=bool, default=True, 
                        help='Nodes transform for Qm9 dataset')
    parser.add_argument('--edge_transform', type=bool, default=True, 
                        help='Edge transform for Qm9 dataset')
    parser.add_argument('--target_transforme', type=bool, default=False, 
                        help='Target transform for Qm9 dataset')
    parser.add_argument('--e_representation', type=str, default='raw_distance', 
                        help='Representation of edges')

    # training parameters
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Numbers of gpus used for training')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Pre-fetching threads.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                        help='Learning rate', dest='lr')    
    parser.add_argument('--resume', type=str, default='', 
                        help='Path to a checkpoint from witch to resume training')
    parser.add_argument('--valid_split', type=float, default=0.1, 
                        help='Percent of the data that is used as validation (0-1)')

    # paths
    parser.add_argument('--data_path', type=str, default='../data/qm9/dsgdb9nsd/',
                        help='Dataset dir path')
    parser.add_argument('--weights_path', type=str, default='D:/WORK/PROJECTS/furnace_segm/models/', 
                        help='Weights dir path')

    # loss function parameters
    

    # testing parameters
    parser.add_argument('--model', type=str, default='../models/epoch=395.ckpt',
                        help='Path to trained model')

    return parser.parse_args(args=[])


def train():
    args = get_args()

    model = ecm.EdgeClassifier(args)

    profiler = ecm.pl.profiler.AdvancedProfiler()
    checkpoint_callback = ecm.ModelCheckpoint(filepath=args.weights_path, save_top_k=1, mode='min', save_weights_only=False)

    if args.resume:
        try:
            print('Start training from checkpoint...')
            trainer = ecm.pl.Trainer(gpus=args.n_gpus, resume_from_checkpoint=args.resume, checkpoint_callback=checkpoint_callback, profiler=False)
        except:
            print('Checkpoint doesnt exist')
    else:
        print('Start training from beginning...')
        trainer = ecm.pl.Trainer(gpus=args.n_gpus, max_epochs=args.epochs, auto_lr_find=False, checkpoint_callback=checkpoint_callback, profiler=False)
  
    trainer.fit(model)


def test():
    args = get_args()

    
if __name__ == "__main__":
    train()
    #test()