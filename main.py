import argparse
import os
import scipy.misc
import numpy as np

from ada_rendering import pose2image
import tensorflow as tf
from pdb import set_trace as st

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=350, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=134, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=128, help='then crop to this size')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=False, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50,
                    help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000,
                    help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False,
                    help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True,
                    help='iter into serial image list')
# parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='/local-scratch/cvpr18/dataset/checkpoint/', help='models are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint-debug-352epoch',
                    help='models are saved here')
parser.add_argument('--root_dir', dest='root_dir',
                    default='/local-scratch2/mzhai/cvpr18/fashion-pose2image-batchsize1/', help='root_dir')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
# parser.add_argument('--test_dir', dest='test_dir', default='/local-scratch2/mzhai/ComputeCanada/final_models/final_models/pose2image-batchsize1/test', help='test sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./ours-rerun-20180712', help='test sample are saved here')
parser.add_argument('--vgg_path', dest='vgg_path',
                    default='/local-scratch/dengr/ruizhid_cedar/local-scratch2/mzhai/cvpr18/dataset/data/pre_trained/beta16/imagenet-vgg-verydeep-19.mat',
                    help='path of the pretrained vgg model')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        print("Creating Model...")
        model = pose2image(sess, image_size=args.fine_size, batch_size=args.batch_size,
                           output_size=args.fine_size, dataset_name=args.dataset_name,
                           checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir, vgg_path=args.vgg_path)
        print("Model Created...")
        # st()
        if args.phase == 'train':
            print("Start to train model...")
            model.train(args)
        else:
            print("Start to test model...")
            model.test(args)


if __name__ == '__main__':
    tf.app.run()
