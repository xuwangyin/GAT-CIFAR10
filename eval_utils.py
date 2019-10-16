import os
import numpy as np
import tensorflow as tf
from model import Model

logit_threshs = np.linspace(-300., 30.0, 1000)

eps8_attack_config = {
    'epsilon': 8.0,
    'num_steps': 20,
    'step_size': 2.0,
    'random_start': False,
    'norm': 'Linf'
}


def batched_run(f, x_placeholder, x, sess):
    batch_size = 1000
    results = []
    for i in range(0, x.shape[0], batch_size):
        results.append(
            sess.run(f, feed_dict={x_placeholder: x[i:i + batch_size]}))
    return np.concatenate(results)


class BaseDetectorFactory:
    def __init__(self):
        self.__checkpoints = []
        self.__base_detectors = []
        self.__detector_savers = []
        self.num_classes = 10
        ckpt_dir = 'models/cifar10_ovr_Linf_8.0_iter40_lr0.5_bs300/'
        for i in range(self.num_classes):
            scope = 'detector-class{}'.format(i)
            self.__base_detectors.append(
                Model(mode='eval', var_scope=scope, target_class=i))
            detector_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=scope)
            self.__detector_savers.append(
                tf.train.Saver(var_list=detector_vars))
            self.__checkpoints.append(
                tf.train.latest_checkpoint(
                    os.path.join(ckpt_dir, 'class{}_ckpt_best'.format(i))))
        self.restored = False

    def restore_base_detectors(self, sess):
        for i in range(self.num_classes):
            self.__detector_savers[i].restore(sess, self.__checkpoints[i])
        self.restored = True

    def get_base_detectors(self):
        return self.__base_detectors


def get_det_logits(x, x_preds, detectors, sess):
    """Compute detector logits for the input.

    First assign x to detectors based on the classifier output (x_preds), 
    then computes detector logit outputs.  
    """
    assert x.shape[0] == x_preds.shape[0]
    det_logits = np.zeros_like(x_preds)
    for classidx in range(10):
        assign = x_preds == classidx
        det_logits[assign] = batched_run(detectors[classidx].target_logits,
                                         detectors[classidx].x_input,
                                         x[assign], sess)
    return det_logits


def get_tpr(x_nat, ths, naive_classifier, detectors, sess):
    """Recall on the set of original data set"""
    nat_preds = batched_run(naive_classifier.predictions,
                            naive_classifier.x_input, x_nat, sess)
    det_logits = get_det_logits(x_nat, nat_preds, detectors, sess)
    tpr = [(det_logits > th).mean() for th in ths]
    return tpr


def get_nat_accs(x_nat, y, ths, naive_classifier, detectors, sess):
    """Accuracy on the natural data set"""
    nat_preds = batched_run(naive_classifier.predictions,
                            naive_classifier.x_input, x_nat, sess)
    det_logits = get_det_logits(x_nat, nat_preds, detectors, sess)
    accs = [(np.logical_and(det_logits > th, nat_preds == y)).mean()
            for th in ths]
    return accs


def get_fpr(x_adv, y, ths, naive_classifier, detectors, sess):
    """The portion of perturbed data samples that are adversarial (adv_preds != y) and
  at the same time successfully fool the detectors (det_logits > th)"""
    adv_preds = batched_run(naive_classifier.predictions,
                            naive_classifier.x_input, x_adv, sess)
    det_logits = get_det_logits(x_adv, adv_preds, detectors, sess)
    fpr = [
        np.logical_and(det_logits > th, adv_preds != y).mean() for th in ths
    ]
    return fpr


def get_adv_errors(x_adv, y, ths, naive_classifier, detectors, sess):
    """With reject option, the naive classifier's error rate on perturbed test set.

  The error rate is computed as the portion of samples that are
  not rejected (det_logits > th) and at the same time
  causing misclassification (adv_preds != y)
  In other words, any samples that are rejected or
  corrected classified, are assumed to be properly handled.
  """
    return get_fpr(x_adv, y, ths, naive_classifier, detectors, sess)
