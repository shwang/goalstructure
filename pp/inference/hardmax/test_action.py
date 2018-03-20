from __future__ import division
from unittest import TestCase
import numpy as np
from numpy import testing as t

from ...mdp import GridWorldMDP
from .action import *

class TestInferJoint(TestCase):
    def test_classic_nocrash(self):
        g = GridWorldMDP(10, 10)
        coor = g.coor_to_state
        dests = [coor(7, 7), coor(2, 2), coor(5, 0)]
        betas = [1, 2, 3, 4]
        T = 5
        traj = [(coor(1, 1), g.Actions.RIGHT), (coor(2, 1), g.Actions.UP)]
        infer_joint(g=g, dests=dests, betas=betas, T=T,
                use_gridless=False, traj=traj)

    def test_gridless_nocrash(self):
        g = GridWorldExpanded(10, 10)
        coor = g.coor_to_state
        dests = [coor(7, 7), coor(2, 2), coor(5, 0)]
        betas = [1, 2, 3, 4]
        T = 5
        traj = [(1, 1), (2, 1), (3, 1), (4.5, 6)]
        infer_joint(g=g, dests=dests, betas=betas, T=T, use_gridless=True,
                traj=traj)

    def test_full_rational(self):
        g = GridWorldMDP(5, 1)
        coor = g.coor_to_state
        res, _, _ = infer_joint(g, dests=[coor(4, 0)], betas=[1e-3], T=5,
                init_state=coor(1,0))

        expect_r = np.zeros(len(g.Actions))
        expect_r[g.Actions.RIGHT] = 1
        expect_a = np.zeros(len(g.Actions))
        expect_a[g.Actions.ABSORB] = 1

        t.assert_allclose(res[0], expect_r)
        t.assert_allclose(res[1], expect_r)
        t.assert_allclose(res[2], expect_r)
        t.assert_allclose(res[3], expect_a)
        t.assert_allclose(res[4], expect_a)

    def test_rational_vs_not(self):
        g = GridWorldMDP(5, 1)
        coor = g.coor_to_state
        res0, _, _ = infer_joint(g, dests=[coor(4, 0)], betas=[1e-3], T=5,
                init_state=coor(1,0))

        res1, _, _ = infer_joint(g, dests=[coor(4, 0)], betas=[1], T=5,
                init_state=coor(1,0))

        A = g.Actions
        self.assertLessEqual(res0[0][A.LEFT], res1[0][A.LEFT])

        self.assertGreaterEqual(res0[0][A.RIGHT], res1[0][A.RIGHT])
        self.assertGreaterEqual(res0[1][A.RIGHT], res1[1][A.RIGHT])
        self.assertGreaterEqual(res0[2][A.RIGHT], res1[2][A.RIGHT])
        self.assertGreaterEqual(res0[3][A.ABSORB], res1[3][A.ABSORB])
        self.assertGreaterEqual(res0[4][A.ABSORB], res1[4][A.ABSORB])

    def test_rational_vs_less_rationals(self):
        g = GridWorldMDP(5, 1)
        coor = g.coor_to_state
        res0, _, _ = infer_joint(g, dests=[coor(4, 0)], betas=[0.1], T=5,
                init_state=coor(1,0))

        res1, _, _ = infer_joint(g, dests=[coor(4, 0)], betas=[0.1, 0.3], T=5,
                init_state=coor(1,0))

        A = g.Actions
        self.assertGreaterEqual(res0[0][A.RIGHT], res1[0][A.RIGHT])
        self.assertGreaterEqual(res0[1][A.RIGHT], res1[1][A.RIGHT])
        self.assertGreaterEqual(res0[2][A.RIGHT], res1[2][A.RIGHT])
        self.assertGreaterEqual(res0[3][A.ABSORB], res1[3][A.ABSORB])
        self.assertGreaterEqual(res0[4][A.ABSORB], res1[4][A.ABSORB])
