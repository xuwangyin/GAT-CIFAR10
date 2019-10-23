import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10_input
from pgd_attack import PGDAttackDetector
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('checkpoint')
parser.add_argument('--prefixed', action='store_true')
parser.add_argument('--rows', type=int, default=1)
parser.add_argument('--cols', type=int, default=20)
parser.add_argument('--norm', type=str, choices=['Linf', 'L2'], default='L2')
parser.add_argument('--epsilon', type=float, default=30*255)
parser.add_argument('--num_steps', type=int, default=60)
parser.add_argument('--step_size', type=float, default=0.5*255)
parser.add_argument('--save_as', type=str)
args = parser.parse_args()

np.random.seed(123)

if args.prefixed:
    detector_var_scope = 'detector-class{}'.format(args.target_class)
else:
    detector_var_scope = 'detector'
detector = Model(mode='eval',
                 var_scope=detector_var_scope,
                 target_class=args.target_class)

detector_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=detector_var_scope)
detector_saver = tf.train.Saver(var_list=detector_vars)

attack_config = {
    'epsilon': args.epsilon,
    'num_steps': args.num_steps,
    'step_size': args.step_size,
    'random_start': False,
    'norm': args.norm
}

attack = PGDAttackDetector(detector, loss_func='cw', **attack_config)
cifar = cifar10_input.CIFAR10Data('cifar10_data')

with tf.Session() as sess:
    detector_saver.restore(sess, args.checkpoint)

    # Estimate Gaussian distribution
    # Based on https://github.com/MadryLab/robustness_applications/blob/master/generation.ipynb
    x_train = cifar.train_data.xs.astype(np.float32) / 255.0
    y_train = cifar.train_data.ys
    x_test_target = x_train[y_train == args.target_class]
    x_test_target = np.reshape(x_test_target, (x_test_target.shape[0], -1))
    mean = np.mean(x_test_target, axis=0)
    flat = x_test_target - mean[np.newaxis, :]
    cov = np.dot(flat.T, flat) / x_test_target.shape[0]
    seeds = np.random.multivariate_normal(mean=mean,
                                          cov=cov + 1e-4 * np.eye(3 * 32 * 32),
                                          size=args.cols*args.rows)
    seeds = seeds.clip(0, 1).reshape((args.cols*args.rows, 32, 32, 3)) * 255.0
    # Perturb seeds
    synthesized = attack.perturb(seeds, np.zeros(seeds.shape[0]), sess)
    # Show Perturbed
    dim, pad = 32, 1
    space = dim + pad
    fig, ax = plt.subplots(1, 1, figsize=(args.cols, args.rows), dpi=100)
    tiling = np.ones((space * args.rows, space * args.cols, 3), dtype=np.float32) * 255
    for row in range(args.rows):
        for col in range(args.cols):
            tiling[row * space:row * space + dim, col * space:col * space +
                   dim] = synthesized[row * args.cols + col]
    ax.imshow(tiling / 255.0)
    ax.axis('off')
    if args.save_as:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace = 0.05, wspace = 0.05)
        plt.savefig(args.save_as, dpi=100)
    plt.show()
