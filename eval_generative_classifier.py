import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10_input
from pgd_attack import PGDAttackClassifier, PGDAttackDetector
from model import Model, BayesClassifier
from eval_utils import BaseDetectorFactory, batched_run, eps8_attack_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

robust_classifier = Model(mode='eval', var_scope='classifier')
classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars)
classifier_checkpoint = 'models/adv_trained_prefixed_classifier/checkpoint-70000'

factory = BaseDetectorFactory()

cifar = cifar10_input.CIFAR10Data('cifar10_data')

num_eval_examples = 10000 if len(sys.argv) <= 1 else int(sys.argv[1])
eval_data = cifar.eval_data
x_test = eval_data.xs.astype(np.float32)[:num_eval_examples]
y_test = eval_data.ys.astype(np.int32)[:num_eval_examples]

plt.figure(figsize=(3.5 * 1.5, 2 * 1.5))

with tf.Session() as sess:
    classifier_saver.restore(sess, classifier_checkpoint)
    factory.restore_base_detectors(sess)

    base_detectors = factory.get_base_detectors()
    bayes_classifier = BayesClassifier(base_detectors)

    # Test robust classifier
    # Natural acc
    preds = batched_run(robust_classifier.predictions,
                        robust_classifier.x_input, x_test, sess)
    print('robust classifier standard acc {}'.format(
        (preds == y_test).mean()))  # nat acc 0.8725

    # Adv acc
    attack = PGDAttackClassifier(classifier=robust_classifier,
                                 loss_func='cw',
                                 **eps8_attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    preds = batched_run(robust_classifier.predictions,
                        robust_classifier.x_input, x_test_adv, sess)
    print('robust classifier adv acc {}, eps=8'.format(
        (preds == y_test).mean()))  # adv acc 0.4689

    # Test generative classifier
    nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    for loss_fn in ['cw', 'xent']:
        attack = PGDAttackClassifier(classifier=bayes_classifier,
                                     loss_func=loss_fn,
                                     **eps8_attack_config)
        x_test_adv = attack.batched_perturb(x_test, y_test, sess)
        adv_errors = bayes_classifier.adv_errors(x_test_adv, y_test, sess)
        if loss_fn == 'cw':
            plt.plot(adv_errors, nat_accs, label='Generative classifier')
        else:
            plt.plot(adv_errors, nat_accs, label='Generative classifier (xent loss)')

    plt.xlabel('Error on perturbed CIFAR10 test set')
    plt.ylabel('Accuracy on CIFAR10 test set')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
