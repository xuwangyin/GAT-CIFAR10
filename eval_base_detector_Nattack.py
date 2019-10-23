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
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--step_size', type=float, default=0.5)
parser.add_argument('--epsilon', type=float, default=8.0)
parser.add_argument('--norm', type=str, choices=['Linf', 'L2'], default='Linf')
parser.add_argument('--prefixed', action='store_true')
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
eval_batch_size = 1
x_test = eval_data.xs.astype(np.float32)[:num_eval_examples]
y_test = eval_data.ys.astype(np.int32)[:num_eval_examples]
x_test_others = x_test[y_test != args.target_class]

npop = 300  # population size
sigma = 0.1  # noise standard deviation
alpha = 0.008  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.
epsi = args.epsilon / 255.

with tf.Session() as sess:
    sess = tf.Session()
    detector_saver.restore(sess, args.checkpoint)
    PGD_logit_outs, Nattack_logit_outs = [], []
    for i in range(100):
        # 1 sample per batch
        x_batch_others = x_test_others[i:i + 1]

        # PGD attack
        x_batch_others_adv = attack.perturb(x_batch_others,
                                            np.zeros(x_batch_others.shape[0]),
                                            sess,
                                            verbose=False)
        feed_dict = {detector.x_input: x_batch_others_adv}
        batch_logits = sess.run(detector.target_logits, feed_dict=feed_dict)
        print('Sample {} PGD attack logit out {}'.format(i, batch_logits[0]))
        PGD_logit_outs.append(batch_logits[0])

        # Nattack, based on https://github.com/Cold-Winter/Nattack/blob/master/therm-adv/re_li_attack.py
        # Clip a_max to prevent np.arctanh overflow 
        img0 = np.clip(x_batch_others / 255, 0, 1 - 1e-7)
        img_shape = img0.shape[1:]
        img0_atan = np.arctanh((img0 - boxplus) / boxmul)
        modify = np.random.randn(1, *img_shape) * 0.001
        running_max = []
        for step in range(1000):
            Nsample = np.random.randn(npop, *img_shape)
            modify_try = modify.repeat(npop, 0) + sigma * Nsample
            img1 = np.tanh(img0_atan + modify_try) * boxmul + boxplus
            img1_proj = img0 + np.clip(img1 - img0, -epsi, epsi)

            feed_dict = {detector.x_input: img1_proj * 255}
            logit_outs = sess.run(detector.target_logits, feed_dict=feed_dict)
            A = (logit_outs - np.mean(logit_outs)) / (np.std(logit_outs) + 1e-7)
            modify = modify + (alpha / (npop * sigma)) * (
                (np.dot(Nsample.reshape(npop, -1).T, A)).reshape(*img_shape))
            # print('iter {} max logit out {}'.format(step, np.max(logit_outs)))
            running_max.append(np.max(logit_outs))
        print('Sample {} Nattack logit out {}\n'.format(i, np.max(running_max)))
        Nattack_logit_outs.append(np.max(running_max))
    print('PGD attack mean {}'.format(np.mean(PGD_logit_outs)))
    print('Nattack mean {}'.format(np.mean(Nattack_logit_outs)))
