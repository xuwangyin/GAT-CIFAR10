import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10_input
from pgd_attack import PGDAttackCombined, PGDAttackClassifier
from model import Model, BayesClassifier
from eval_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

classifier = Model(mode='eval', var_scope='classifier')
classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars)
classifier_checkpoint = 'models/naturally_trained_prefixed_classifier/checkpoint-70000'

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

    # TPR on natural set for integrated detection
    nat_tpr = get_tpr(x_test, logit_threshs, classifier, base_detectors, sess)

    # Integrated detection - classifier attack
    attack = PGDAttackClassifier(classifier, loss_func='cw', **eps8_attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    adv_fpr = get_fpr(x_test_adv, y_test, logit_threshs, classifier, base_detectors, sess)
    plt.plot(adv_fpr, nat_tpr, label='Integrated detection (classifier attack)')
    
    # Integrated detection - detector attack
    attack = PGDAttackClassifier(bayes_classifier, loss_func='cw', **eps8_attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    adv_fpr = get_fpr(x_test_adv, y_test, logit_threshs, classifier, base_detectors, sess)
    plt.plot(adv_fpr, nat_tpr, label='Integrated detection (detector attack)')

    # Integrated detection - combined attack
    for loss_fn in ['default', 'cw']:
        attack = PGDAttackCombined(classifier, bayes_classifier, loss_fn=loss_fn, **eps8_attack_config)
        x_test_adv = attack.batched_perturb(x_test, y_test, sess)
        adv_fpr = get_fpr(x_test_adv, y_test, logit_threshs, classifier, base_detectors, sess)
        if loss_fn == 'cw':
            plt.plot(adv_fpr, nat_tpr, label='Integrated detection (combined attack cw loss)')
        else:
            plt.plot(adv_fpr, nat_tpr, label='Integrated detection (combined attack)')

    # TPR on natural set for generative detection
    nat_tpr = bayes_classifier.nat_tpr(x_test, sess)

    # Generative detection
    attack = PGDAttackClassifier(bayes_classifier, loss_func='cw', **eps8_attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    adv_fpr = bayes_classifier.adv_fpr(x_test_adv, y_test, sess)
    plt.plot(adv_fpr, nat_tpr, label='Generative detection')

    # Generative detection - cross-entropy loss
    attack = PGDAttackClassifier(bayes_classifier, loss_func='xent', **eps8_attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    adv_fpr = bayes_classifier.adv_fpr(x_test_adv, y_test, sess)
    plt.plot(adv_fpr, nat_tpr, label='Generative detection (cross-entropy loss)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
