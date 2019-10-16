import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from model import BayesClassifier


class PGDAttack:
  def __init__(self, epsilon, num_steps, step_size, random_start, norm):
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    self.norm = norm
    assert norm in ['Linf', 'L2']

  def perturb(self, x_nat, y, sess, c_constants=None, verbose=False):
    """Based on https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py"""
    if self.rand:
      if self.norm == 'L2':
        delta = np.random.randn(*x_nat.shape)
        scale = np.random.uniform(low=0.0, high=self.epsilon, size=[delta.shape[0], 1, 1, 1])
        delta = scale * delta / self._l2_norm(delta)
        x = x_nat + delta
      else:
        x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      # x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = np.copy(x_nat).astype(np.float32)
      #x = np.copy(x_nat)

    if verbose:
      plt.figure(figsize=(10, 10))
    for i in range(self.num_steps):
      if verbose:
        # loss, logits = sess.run([self.model.y_xent, self.model.target_logits],
        #                         feed_dict={self.model.x_input: x, self.model.y_input: y})
        loss, logits = sess.run([self.model.y_xent, self.model.logits],
                                feed_dict={self.model.x_input: x, self.model.y_input: y})
        logits = logits[np.arange(logits.shape[0]), y]

        if self.norm == 'L2':
          dist = np.squeeze(self._l2_norm(x - x_nat))
        else:
          dist = np.max(np.abs(x - x_nat), axis=(1, 2, 3))
        print('step {}'.format(i), end=': logits ')
        print(np.round(logits[:5], 3), end='| loss ')
        print(np.round(loss[:5], 3), end='| dist ')
        print(np.round(dist[:5], 3))
        if i % 10 == 0:
          # M = N = int(np.sqrt(x.shape[0]))
          ncols = 10
          nrows = 5
          dim = 32
          pad = 1
          space = dim + pad
          tiling = np.zeros((space * nrows, space * ncols, 3), dtype=np.float32) + 255.0
          for row in range(nrows):
            for col in range(ncols):
              tiling[row*space: row*space+dim, col*space: col*space+dim] = x[row*ncols+col]
          plt.imshow(tiling/255.0)
          plt.axis('off')
          #plt.show()
          plt.savefig('painting/img{:04d}.pdf'.format(i//5), bbox_inches='tight', dpi=500)
      if y is None:
        grad = sess.run(self.grad, feed_dict={self.x_input: x})
      else:
        grad = sess.run(self.grad, feed_dict={self.x_input: x, self.y_input: y})
      # note here we are performing gradient ascent
      if c_constants is not None:
        grad = grad - c_constants[:, None, None, None] * 2 * (x - x_nat)
      if self.norm == 'Linf':
        x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')
        x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      else:
        grad_norm = self._l2_norm(grad)
        grad_norm = np.clip(grad_norm, a_min=np.finfo(float).eps, a_max=None)
        x = np.add(x, self.step_size * grad / grad_norm, out=x, casting='unsafe')
        dx = x - x_nat
        dx_norm = self._l2_norm(dx)
        dx_final_norm = dx_norm.clip(0, self.epsilon)
        dx_norm = np.clip(dx_norm, a_min=np.finfo(float).eps, a_max=None)
        x = x_nat + dx_final_norm * dx / dx_norm
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x
 
  def batched_perturb(self, x, y, sess, batch_size=50):
    x_adv = []
    for i in range(0, x.shape[0], batch_size):
      print('perturbed {}-{}'.format(i, i + batch_size))
      x_batch = x[i: i + batch_size]
      y_batch = y[i: i + batch_size]
      x_adv.append(self.perturb(x_batch, y_batch, sess))
    return np.concatenate(x_adv)


  @staticmethod
  def _l2_norm(batch):
    return np.sqrt(np.sum(batch ** 2, axis=(1, 2, 3), keepdims=True))

class PGDAttackDetector(PGDAttack):
  def __init__(self, detector, loss_func, **kwargs):
    super().__init__(**kwargs)
    self.model = detector
    self.x_input = detector.x_input
    self.y_input = detector.y_input
    if loss_func == 'xent':
      loss = detector.xent
    elif loss_func == 'cw':
      loss = detector.target_logits
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = detector.xent

    self.grad = tf.gradients(loss, detector.x_input)[0]

class PGDAttackClassifier(PGDAttack):
  def __init__(self, classifier, loss_func, targeted=False, **kwargs):
    super().__init__(**kwargs)
    self.x_input = classifier.x_input
    self.y_input = classifier.y_input
    self.model = classifier

    if loss_func == 'xent':
      loss = classifier.xent
      if targeted:
        loss = -classifier.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(classifier.y_input, 10, dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * classifier.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * classifier.pre_softmax - 1e4*label_mask, axis=1)
      if isinstance(classifier, BayesClassifier):
        loss = -tf.reduce_sum(-wrong_logit)
      else:
        loss = -tf.reduce_sum(correct_logit - wrong_logit)
      if targeted:
        loss = tf.reduce_sum(correct_logit)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = classifier.xent

    self.grad = tf.gradients(loss, classifier.x_input)[0]


class PGDAttackCombined(PGDAttack):
  def __init__(self, naive_classifier, bayes_classifier, loss_fn, **kwargs):
    super().__init__(**kwargs)

    assert loss_fn in ['cw', 'default']
    assert isinstance(bayes_classifier, BayesClassifier)

    self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x_input')
    self.y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')
    clf_logits = naive_classifier.forward(self.x_input)
    det_logits = bayes_classifier.forward(self.x_input)

    label_mask = tf.one_hot(self.y_input, 10, dtype=tf.float32)
    clf_correct_logit = tf.reduce_sum(label_mask * clf_logits, axis=1)
    clf_wrong_logit = tf.reduce_max((1 - label_mask) * clf_logits - 1e4 * label_mask, axis=1)
    det_wrong_logit = tf.reduce_max((1 - label_mask) * det_logits - 1e4 * label_mask, axis=1)

    if loss_fn == 'cw':
      with_det_logits = (-det_wrong_logit + 1) * tf.reduce_max(clf_logits, axis=1)
      correct_logit_with_det = tf.maximum(clf_correct_logit, with_det_logits)
      self.loss = -tf.reduce_sum(correct_logit_with_det - clf_wrong_logit)
    else:
      mask = tf.cast(tf.greater(clf_wrong_logit, clf_correct_logit), tf.float32)
      self.loss = -tf.reduce_sum(mask * (-det_wrong_logit) + (1.0 - mask) * (clf_correct_logit - clf_wrong_logit))

    self.grad = tf.gradients(self.loss, self.x_input)[0]


