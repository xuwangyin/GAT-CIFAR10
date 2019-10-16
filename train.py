import json
from datetime import datetime
import os
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from model import Model
import cifar10_input
from pgd_attack import PGDAttackDetector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('--norm', type=str, choices=['Linf', 'L2'], default='Linf')
parser.add_argument('--epsilon', type=float, default=8.0)
parser.add_argument('--num_steps', type=int, default=40)
parser.add_argument('--step_size', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--prefixed', action='store_true')
parser.add_argument( '--pretrained', type=str,
    default='models/naturally_trained_prefixed_detector/checkpoint-70000')
args = parser.parse_args()
print(args)

# Settings follow
# https://github.com/MadryLab/cifar10_challenge/blob/master/config.json
tf.set_random_seed(451760341)
np.random.seed(216105420)
max_num_training_steps = 40000
num_checkpoint_steps = 500

raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")
global_step = tf.contrib.framework.get_or_create_global_step()

if args.prefixed:
    var_scope = 'detector-class{}'.format(args.target_class)
else:
    var_scope = 'detector'
# Set model to 'eval' to disable batch normalization
model = Model(mode='eval', var_scope=var_scope, target_class=args.target_class)

opt = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9)
total_loss = model.mean_xent + 2e-4 * model.weight_decay_loss
train_step = opt.minimize(total_loss, global_step=global_step)

attack = PGDAttackDetector(model,
                           loss_func='xent',
                           epsilon=args.epsilon,
                           num_steps=args.num_steps,
                           step_size=args.step_size,
                           random_start=True,
                           norm=args.norm)

model_dir = 'models/cifar10_ovr_{}_{}_iter{}_lr{}_bs{}'.format(
    args.norm, args.epsilon, args.num_steps, args.step_size, args.batch_size)
Path(model_dir).mkdir(parents=True, exist_ok=True)

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)
saver = tf.train.Saver(var_list=var_list, max_to_keep=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

    # Load pretrained model
    saver.restore(sess, args.pretrained)

    x_test = cifar.eval_data.raw_datasubset.xs.astype(np.float32)
    y_test = cifar.eval_data.raw_datasubset.ys.astype(np.int64)
    x_test_target = x_test[y_test == args.target_class]
    x_test_others = x_test[y_test != args.target_class]

    x_test_with_nat = np.concatenate([x_test_target, x_test_others])
    y_test_with_nat = np.concatenate([
        np.ones(x_test_target.shape[0], dtype=np.int64),
        np.zeros(x_test_others.shape[0], dtype=np.int64)
    ])
    nat_test_dict = {
        model.x_input: x_test_with_nat,
        model.y_input: y_test_with_nat
    }

    for step in range(0, max_num_training_steps + 1):
        try:
            x_batch, y_batch = cifar.train_data.get_next_batch(
                args.batch_size, args.target_class, multiple_passes=True)
        except ValueError:
            continue

        x_batch_target = x_batch[y_batch == args.target_class]
        x_batch_others = x_batch[y_batch != args.target_class]
        y_batch_target = y_batch[y_batch == args.target_class]
        y_batch_others = y_batch[y_batch != args.target_class]

        x_batch_others_adv = attack.perturb(
            x_batch_others, np.zeros(x_batch_others.shape[0], dtype=np.int64),
            sess)
        x_batch_adv = np.concatenate([x_batch_target, x_batch_others_adv])

        x_batch = np.concatenate([x_batch_target, x_batch_others])

        y_batch = np.concatenate([
            np.ones(x_batch_target.shape[0], dtype=np.int64),
            np.zeros(x_batch_others.shape[0], dtype=np.int64)
        ])

        nat_dict = {model.x_input: x_batch, model.y_input: y_batch}

        adv_dict = {model.x_input: x_batch_adv, model.y_input: y_batch}

        logits, acc, tpr, fpr = sess.run([
            model.target_logits, model.accuracy, model.true_positive_rate,
            model.false_positive_rate
        ],
                                         feed_dict=adv_dict)
        fpr_, tpr_, thresholds = roc_curve(y_batch, logits)

        print('Step {}: ({}),'.format(step, datetime.now()), end=' ')

        print('Training adv auc {:.4f}, acc {:.4f}, tpr {:.4f}, fpr {:.4f},'.
              format(auc(fpr_, tpr_), acc, tpr, fpr),
              end=' ')

        print('pos {}, neg {}'.format(x_batch_target.shape[0],
                                      x_batch_others.shape[0]))

        # Save checkpoint
        if step % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(model_dir,
                                    'class{}'.format(args.target_class),
                                    'checkpoint'),
                       global_step=step)
            logits = sess.run(model.target_logits, feed_dict=nat_test_dict)
            fpr_, tpr_, thresholds = roc_curve(y_test_with_nat, logits)

            print('Step {}: ({}),'.format(step, datetime.now()), end=' ')

            print('Test nat auc {:.4f},'.format(auc(fpr_, tpr_)), end=' ')

            print('pos {}, neg {}'.format(x_test_target.shape[0],
                                          x_test_others.shape[0]))

        # Train the model
        sess.run(train_step, feed_dict=adv_dict)
