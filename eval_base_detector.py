import os
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import cifar10_input
from pgd_attack import PGDAttackDetector
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('checkpoint')
parser.add_argument('--steps', type=int, default=40)
parser.add_argument('--step_size', type=float, default=0.5)
parser.add_argument('--epsilon', type=float, default=8.0)
parser.add_argument('--norm', type=str, choices=['Linf', 'L2'], default='Linf')
parser.add_argument('--prefixed', action='store_true')
args = parser.parse_args()

np.random.seed(123)

classifier = Model(mode='eval', var_scope='classifier')
classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars)
classifier_checkpoint = 'models/naturally_trained_prefixed_classifier/checkpoint-70000'

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

attack = PGDAttackDetector(detector,
                           loss_func='cw',
                           epsilon=args.epsilon,
                           num_steps=args.steps,
                           step_size=args.step_size,
                           random_start=False,
                           norm=args.norm)
print('using checkpoint {}'.format(args.checkpoint))
cifar = cifar10_input.CIFAR10Data('cifar10_data')
eval_data = cifar.eval_data

num_eval_examples = 10000
eval_batch_size = 200
x_test = eval_data.xs.astype(np.float32)[:num_eval_examples]
y_test = eval_data.ys.astype(np.int32)[:num_eval_examples]


with tf.Session() as sess:
    classifier_saver.restore(sess, classifier_checkpoint)
    detector_saver.restore(sess, args.checkpoint)

    y_pred = []
    for i in range(0, num_eval_examples, eval_batch_size):
        feed_dict = {classifier.x_input: x_test[i:i + eval_batch_size]}
        y_pred.append(sess.run(classifier.predictions, feed_dict=feed_dict))
    y_pred = np.concatenate(y_pred)
    print('Nat trained classifier accuracy {}'.format(
        (y_pred == y_test).mean()))

    # Create test dataset
    target_mask = y_pred == args.target_class  # D^f_k
    others_mask = np.logical_and(y_test != args.target_class,
                                 y_pred != args.target_class)  # D^f_\k
    # target_mask = y_test == args.target_class
    # others_mask = y_test != args.target_class

    x_test_target = x_test[target_mask]
    y_test_target = y_test[target_mask]
    x_test_others = x_test[others_mask]
    y_test_others = y_test[others_mask]
    x_test = np.concatenate([x_test_target, x_test_others])
    y_test = np.concatenate([y_test_target, y_test_others])

    # Create perturbed dataset
    x_test_others_adv = []
    for i in range(0, num_eval_examples, eval_batch_size):
        x_batch = x_test[i:i + eval_batch_size]
        y_batch = y_test[i:i + eval_batch_size]

        x_batch_target = x_batch[y_batch == args.target_class]
        x_batch_others = x_batch[y_batch != args.target_class]
        if x_batch_others.shape[0] == 0:
            continue
        x_batch_others_adv = attack.perturb(x_batch_others,
                                            np.zeros(x_batch_others.shape[0]),
                                            sess,
                                            verbose=False)
        x_test_others_adv.append(x_batch_others_adv)
    x_test_others_adv = np.concatenate(x_test_others_adv)

    # Natural samples -> 1, perturbed samples -> 0
    x_test_with_adv = np.concatenate([x_test_target, x_test_others_adv])
    y_test_with_adv = np.concatenate([
        np.ones(x_test_target.shape[0], dtype=np.int32),
        np.zeros(x_test_others_adv.shape[0], dtype=np.int32)
    ])

    # Compute detection AUC
    logits = []
    for i in range(0, x_test_with_adv.shape[0], eval_batch_size):
        x_batch = x_test_with_adv[i:i + eval_batch_size]
        y_batch = y_test_with_adv[i:i + eval_batch_size]
        feed_dict = {detector.x_input: x_batch, detector.y_input: y_batch}
        batch_logits = sess.run(detector.target_logits,
                                feed_dict=feed_dict)
        logits.append(batch_logits)
    logits = np.concatenate(logits)

    fpr_, tpr_, thresholds = roc_curve(y_test_with_adv, logits)
    print('Test Adv AUC {}'.format(np.round(auc(fpr_, tpr_), 5)))
