import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from train import trainer
from test import test

def main(opt):
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
        
    if opt.mode == 'train':
        trainer(opt)

    elif opt.mode == 'test':
        test(opt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='LPDGAN', type=str)
    parser.add_argument('--dataroot', type=str, default=r'')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--ndf', type=int, default=64)

    # Train
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--load_iter', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=10400)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--save_latest_freq', type=int, default=5000)
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument('--save_by_iter', action='store_true')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--gan_mode', type=str, default='wgangp')
    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    # Test
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    parser.add_argument('--num_test', type=int, default=1000, help='how many test images to run')

    # For display
    parser.add_argument('--display_freq', type=int, default=10400, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=3,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--update_html_freq', type=int, default=1000,
                        help='frequency of saving training results to html')
    parser.add_argument('--no_html', action='store_true',
                        help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

    args = parser.parse_args()
    print(args)
    main(args)


