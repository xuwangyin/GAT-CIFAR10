import os
import numpy as np
import tensorflow as tf
import cifar10_input
from pgd_attack import PGDAttackClassifier, PGDAttackDetector
from model import Model, BayesClassifier
from eval_utils import BaseDetectorFactory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)


def update_bound(lower, upper, current, success):
    if success:
        lower = current
    else:
        upper = current
    return lower, upper, (lower + upper) * 0.5


attack_config = {
    'epsilon': 256 * 11,
    'num_steps': 100,
    'step_size': 0.1 * 256,
    'random_start': False,
    'norm': 'L2'
}

thresh = -5

factory = BaseDetectorFactory()

cifar = cifar10_input.CIFAR10Data('cifar10_data')

num_eval_examples = 10000
eval_data = cifar.eval_data
x_test = eval_data.xs.astype(np.float32)[:num_eval_examples]
y_test = eval_data.ys.astype(np.int32)[:num_eval_examples]

with tf.Session() as sess:
    factory.restore_base_detectors(sess)

    # # Validate the threshold gives .95 TPR
    # x_test_logits = bayes_classifier.batched_run(bayes_classifier.logits, x_test, sess)
    # print((np.max(x_test_logits, 1) > thresh).mean())  # 0.9486

    base_detectors = factory.get_base_detectors()
    bayes_classifier = BayesClassifier(base_detectors)

    # Assign attack target for each sample
    targets = np.zeros(x_test.shape[0], dtype=np.int64)
    for i in range(x_test.shape[0]):
        targets[i] = np.random.choice(np.delete(np.arange(10), y_test[i]))

    all_targets_best_dists = []
    for target in range(0, 10):
        detector = base_detectors[target]
        x_test_sub = x_test[targets == target]
        y_test_sub = y_test[targets == target]
        attack = PGDAttackDetector(detector=detector,
                                   loss_func='cw',
                                   **attack_config)

        best_dists, best_adv = [], []

        batch_size = 10
        for b in range(0, x_test_sub.shape[0], batch_size):
            print('target {} processing batch {}-{}'.format(
                target, b, b + batch_size))
            x_batch = x_test_sub[b:b + batch_size]
            y_batch = y_test_sub[b:b + batch_size]
            lowers = np.zeros(x_batch.shape[0])
            uppers = np.zeros(x_batch.shape[0]) + 1

            # # Validate that the upper bound will fail all
            # c_constants = np.zeros(x_batch.shape[0]) + 10
            # x_batch_adv = attack.perturb(x_batch, y_batch, sess, c_constants=c_constants)
            # adv_logits = sess.run(bayes_classifier.logits, feed_dict={bayes_classifier.x_input: x_batch_adv})
            # success = np.logical_and(np.argmax(adv_logits, 1) == target, np.max(adv_logits, 1) > thresh)
            # print('success {}'.format(success.mean()))
            # continue

            c_constants = np.zeros(x_batch.shape[0])
            batch_best_dists = np.zeros(x_batch.shape[0]) + 1e9
            batch_best_adv = np.zeros_like(x_batch)

            for depth in range(20):
                x_batch_adv = attack.perturb(x_batch,
                                             y_batch,
                                             sess,
                                             c_constants=c_constants)
                adv_logits = sess.run(
                    bayes_classifier.logits,
                    feed_dict={bayes_classifier.x_input: x_batch_adv})
                success = np.logical_and(
                    np.argmax(adv_logits, 1) == target,
                    np.max(adv_logits, 1) > thresh)
                dist = np.linalg.norm(
                    (x_batch_adv - x_batch).reshape(-1, 32 * 32 * 3), axis=1)
                for i in range(adv_logits.shape[0]):
                    # Sample output
                    if i == 1:
                        print(
                            'sample {} depth {} lower {} upper {} c_constant {} dist {} sucess {}'
                            .format(i, depth + 1, lowers[i], uppers[i],
                                    c_constants[i], dist[i], success[i]))
                    # Update bounds
                    lowers[i], uppers[i], c_constants[i] = update_bound(
                        lowers[i], uppers[i], c_constants[i], success[i])
                    if success[i] and dist[i] < batch_best_dists[i]:
                        batch_best_dists[i] = dist[i]
                        batch_best_adv[i] = x_batch_adv[i]
            print('success {}'.format((batch_best_dists < 1e8).mean()))
            print('dist mean: {}'.format(
                batch_best_dists[batch_best_dists < 1e8].mean()))
            best_dists.append(batch_best_dists)
            best_adv.append(batch_best_adv)
        best_dists = np.concatenate(best_dists)
        best_adv = np.concatenate(best_adv)
        # np.savez(os.path.join(min_dist_data_dir,
        #                       'target{}.npz'.format(target)),
        #          x_test_sub=x_test_sub,
        #          y_test_sub=y_test_sub,
        #          best_dists=best_dists,
        #          best_adv=best_adv)
        all_targets_best_dists.append(best_dists)
    all_targets_best_dists = np.concatenate(all_targets_best_dists)
    print('sucess {}, dist mean {}'.format(
        (all_targets_best_dists < 1e8).mean(),
        all_targets_best_dists[all_targets_best_dists < 1e8].mean()))
