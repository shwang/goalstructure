import numpy as np
from numpy import testing as t
from unittest import TestCase
from pp.mdp import GridWorldMDP

import itertools

from syn import *
from layers import *

def build_deterministic_dataset():
    """
    Dataset which always returns all nine of the (state, action) pairs in
    a 3x3 gridworld where the goal is (2, 2).
    """
    g = GridWorldMDP(3, 3)
    coor = g.coor_to_state
    A = g.Actions
    policy = [
        ((0, 0), A.UP_RIGHT), ((1, 0), A.UP_RIGHT), ((2, 0), A.UP),
        ((0, 1), A.UP_RIGHT), ((1, 1), A.UP_RIGHT), ((2, 1), A.UP),
        ((0, 2), A.RIGHT), ((1, 2), A.RIGHT), ((2, 2), A.ABSORB)]
    X = np.array([e[0] for e in policy])
    Y = np.array([e[1] for e in policy])
    Z = np.array([coor(2, 2)] * 9)

    return Data(X, Y, Z, name="tiny deterministic")
dd = build_deterministic_dataset()

class TestDeterministicDataSet(TestCase):
    def test_meta(self):
        data = build_deterministic_dataset()
        for _ in range(10):
            X, Y = data.get_batch(9)
            t.assert_array_equal(np.sort(X, axis=None), np.sort(data.X, axis=None))
            t.assert_array_equal(np.sort(Y), np.sort(data.Y))


class TestTinyPolicyMap(TestCase):
    def test_give_everything(self):
        model = DirectAction()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        model.train_model(sess, dd, mb_size=9, iters=300, test_data=dd)
        model.assess_model(sess, dd)

class TestDirectActions(TestCase):
    def test_no_crash(self):
        model = DirectActions()
        model.experiment()

class TestDirectAuxLabelEnd(TestCase):
    def test_no_crash(self):
        model = DirectAuxLabelEnd(G=2)
        model.experiment()

class TestDirectAuxLabelMid(TestCase):
    def test_no_crash(self):
        model = DirectAuxLabelMid(G=2)
        model.experiment(samples=200)
