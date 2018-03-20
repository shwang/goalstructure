import numpy as np
from numpy import testing as t
from unittest import TestCase
from pp.mdp import GridWorldMDP

import itertools

from syn import *

class TestGenPredictGoal(TestCase):
    def test_no_crash(self):
        g = GridWorldMDP(15, 15)
        goals = [g.coor_to_state(9, 9), g.coor_to_state(1, 1),
                g.coor_to_state(3, 3)]
        data = gen_predict_goal(g, goals=goals, k=5)
        self.assertEqual(data.N, len(data.Y))

class TestGenPredictPolicy(TestCase):
    def test_no_crash(self):
        g = GridWorldMDP(15, 15)
        goals = [g.coor_to_state(9, 9), g.coor_to_state(1, 1),
                g.coor_to_state(3, 3)]
        data = gen_predict_policy(g, goals=goals)
        self.assertEqual(data.N, len(data.Y))

    def test_no_crash2(self):
        g = GridWorldMDP(15, 15)
        goals = [g.coor_to_state(9, 9), g.coor_to_state(1, 1),
                g.coor_to_state(3, 3)]
        data = gen_predict_policy(g, goals=goals, samples=30)
        self.assertEqual(data.N, len(data.Y))

        for y in data.Y:
            self.assertTrue(0 <= y < g.A)
        for z in data.Z:
            self.assertTrue(0 <= z < len(goals))

class TestGenPredictPolicy2(TestCase):
    def test_no_crash(self):
        g = GridWorldMDP(15, 15)
        goals = [g.coor_to_state(9, 9), g.coor_to_state(1, 1),
                g.coor_to_state(3, 3)]
        data = gen_predict_policy2(g, goals=goals)
        self.assertEqual(data.N, len(data.Y))

class TestGenPredictTraj(TestCase):
    def test_no_crash(self):
        g = GridWorldMDP(15, 15)
        goals = [g.coor_to_state(9, 9), g.coor_to_state(1, 1),
                g.coor_to_state(3, 3)]
        data = gen_predict_traj(g, goals=goals, k=3, l=3)
        self.assertEqual(data.N, len(data.Y))

class TestGenPredictActions(TestCase):
    def test_no_crash(self):
        g = GridWorldMDP(15, 15)
        goals = [g.coor_to_state(9, 9), g.coor_to_state(1, 1),
                g.coor_to_state(3, 3)]
        data = gen_predict_actions(g, goals=goals, k=3, l=3)
        self.assertEqual(data.N, len(data.Y))

class TestData(TestCase):
    def test_batch_one(self):
        data = Data([1], [2])
        x, y = data.get_batch(1)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 1)
        self.assertEqual(x, [1])
        self.assertEqual(y, [2])

    def test_batch_twelve(self):
        data = Data(range(20), np.arange(20))
        for i in range(10):
            x, y = data.get_batch(12)
            self.assertEqual(len(x), len(y))
            self.assertEqual(len(x), 12)
            t.assert_equal(x, y)

    def test_batch_twelve_aux(self):
        data = Data(range(20), np.arange(20), np.arange(20))
        for i in range(10):
            x, y, z = data.get_batch(12, enable_aux=True)
            self.assertEqual(len(x), len(y))
            self.assertEqual(len(x), 12)
            t.assert_equal(x, y)
            t.assert_equal(y, z)

class TestPuddlesWorld(TestCase):
    def test_puddles(self):
        g = puddles_world(10, p=0.5)
        import pdb; pdb.set_trace()
