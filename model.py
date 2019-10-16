# Based on https://github.com/MadryLab/cifar10_challenge/blob/master/model.py
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x.astype(np.float64))) 

class Model(object):
  """ResNet model."""

  def __init__(self, mode, var_scope, target_class=None):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self.target_class = target_class
    self.var_scope = var_scope

    self.x_input = tf.placeholder(
      tf.float32,
      shape=[None, 32, 32, 3])

    self.y_input = tf.placeholder(tf.int64, shape=None)
    self.pre_softmax = self.forward(self.x_input)
    self.logits = self.pre_softmax

    if self.target_class is not None:
      self.target_logits = self.pre_softmax[:, self.target_class]
      self.predictions = tf.cast(self.target_logits > 0, tf.int64)
      self.correct_prediction = tf.equal(self.predictions, self.y_input)
      self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
      self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

      true_positives = tf.bitwise.bitwise_and(self.y_input, self.predictions)
      self.true_positive_rate = tf.reduce_sum(true_positives) / tf.reduce_sum(self.y_input)

      false_positives = tf.bitwise.bitwise_and(1 - self.y_input, self.predictions)
      self.false_positive_rate = tf.reduce_sum(false_positives) / tf.reduce_sum(1 - self.y_input)

      self.recall = self.true_positive_rate
      self.precision = tf.reduce_sum(true_positives) / tf.reduce_sum(self.predictions)

      self.f_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)

      self.balanced_accuracy = 0.5 * (self.true_positive_rate + (1.0 - self.false_positive_rate))
    else:
      self.predictions = tf.argmax(self.pre_softmax, 1)
      self.correct_prediction = tf.equal(self.predictions, self.y_input)
      self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
      self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      if self.target_class is not None:
        self.y_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y_input, tf.float32),
                                                              logits=self.target_logits)
      else:
        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

  def forward(self, x_input):
    with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
      return self.build_model(x_input)

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def build_model(self, x_input):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    with tf.variable_scope('input'):



      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               x_input)
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))



    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    filters = [16, 160, 320, 640]


    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, 5):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, 5):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, 5):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      return self._fully_connected(x, 10)

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


class BayesClassifier(object):
  def __init__(self, detectors):
    self.y_input = tf.placeholder(tf.int64, shape=[None])
    self.output_size = 10
    self.detectors = detectors

    self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])


    self.logits = self.forward(self.x_input)
    self.likelihoods = tf.sigmoid(self.logits)
    self.pre_softmax = self.logits

    self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=self.y_input, logits=self.logits)

    self.xent = tf.reduce_sum(self.y_xent)

    self.predictions = tf.argmax(self.logits, 1)

    correct_prediction = tf.equal(self.predictions, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # self.ths = np.linspace(0, 1, 1000)
    # nat logits min-max -282.4098205566406/17.22737693786621
    # adv logits min-max -266.2849426269531/15.494993209838867
    self.logit_ths = np.linspace(-300, 30, 1000)

  def forward(self, x):
    # shape: [batch, num_classes]
    return tf.stack([net.forward(x)[:, net.target_class] for net in self.detectors], axis=1)

  def batched_run(self, f, x, sess):
    batch_size = 1000
    results = []
    for i in range(0, x.shape[0], batch_size):
      results.append(sess.run(f, feed_dict={self.x_input: x[i: i+batch_size]}))
    return np.concatenate(results)

  def nat_accs(self, x_nat, y, sess, cache_dir=None):
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, 'x_test_logits.npy')
        if not os.path.exists(cache_file):
          Path(cache_dir).mkdir(parents=True, exist_ok=True)
          nat_logits = self.batched_run(self.logits, x_nat, sess)
          np.save(cache_file, nat_logits)
        else:
          nat_logits = np.load(cache_file)
    else:
        nat_logits = self.batched_run(self.logits, x_nat, sess)
    nat_preds = np.argmax(nat_logits, axis=1)
    # p_x = np.mean(sigmoid(nat_logits), axis=1)
    p_x = np.max(nat_logits, axis=1)
    nat_accs = [np.logical_and(p_x > th, nat_preds == y).mean() for th in self.logit_ths]
    return nat_accs

  def nat_tpr(self, x_nat, sess, cache_dir=None):
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, 'x_test_logits.npy')
        if not os.path.exists(cache_file):
          Path(cache_dir).mkdir(parents=True, exist_ok=True)
          nat_logits = self.batched_run(self.logits, x_nat, sess)
          np.save(cache_file, nat_logits)
        else:
          nat_logits = np.load(cache_file)
    else:
        nat_logits = self.batched_run(self.logits, x_nat, sess)
    # p_x = np.mean(sigmoid(nat_logits), axis=1)
    p_x = np.max(nat_logits, axis=1)
    nat_tpr = [(p_x > th).mean() for th in self.logit_ths]
    return nat_tpr

  def adv_errors(self, x_adv, y, sess, cache_dir=None):
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, 'x_test_adv_logits.npy')
        if not os.path.exists(cache_file):
          Path(cache_dir).mkdir(parents=True, exist_ok=True)
          adv_logits = self.batched_run(self.logits, x_adv, sess)
          np.save(cache_file, adv_logits)
        else:
          adv_logits = np.load(cache_file)
    else:
        adv_logits = self.batched_run(self.logits, x_adv, sess)
    # print('adv logits min-max {}/{}'.format(adv_logits.min(), adv_logits.max()))
    adv_preds = np.argmax(adv_logits, axis=1)
    # p_x = np.mean(sigmoid(adv_logits), axis=1)
    p_x = np.max(adv_logits, axis=1)
    adv_errors = [np.logical_and(p_x > th, adv_preds != y).mean() for th in self.logit_ths]
    return adv_errors

  def adv_fpr(self, x_adv, y, sess, cache_dir=None):
    return self.adv_errors(x_adv, y, sess, cache_dir)
