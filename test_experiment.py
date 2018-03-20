from unittest import TestCase
from layers import *
from experiment import *
from syn import *

class TestTFModel(TestCase):
    def test_basic(self, N=50, iters=1000, samples=10000):
        g = GridWorldMDP(N, N)
        goals = [g.coor_to_state(N-1, N-1)]

        nn = DirectActions()

        data = syn.gen_predict_actions(g, goals, nn.k, nn.l,
                samples=samples, beta=1e-3)
        test_data = syn.gen_predict_actions(g, goals, nn.k, nn.l,
                samples=100, beta=1e-3)


        model = TFModel(nn)
        model.evaluate(test_data)
        model.train(data)
        model.predict(test_data)
        model.evaluate(test_data)

    def test_basic_aux_end(self, N=50, iters=1000, samples=10000):
        g = GridWorldMDP(N, N)
        goals = [g.coor_to_state(N-1, N-1)]

        nn = DirectAuxLabelEnd(G=len(goals))

        data = syn.gen_predict_actions(g, goals, nn.k, nn.l,
                samples=samples, beta=1e-3)
        test_data = syn.gen_predict_actions(g, goals, nn.k, nn.l,
                samples=100, beta=1e-3)


        model = TFModel(nn)
        model.evaluate(test_data)
        model.train(data)
        model.predict(test_data)
        model.evaluate(test_data)

class TestSSModel(TestCase):
    def test_nocrash(self, N=50, samples=10000):
        g = GridWorldMDP(N, N)
        goals = [g.coor_to_state(N-1, N-1), g.coor_to_state(0, 0)]

        k, l = 3, 3

        data = syn.gen_predict_actions(g, goals, k, l,
                samples=samples, beta=1e-3)
        test_data = syn.gen_predict_actions(g, goals, k, l,
                samples=100, beta=1e-3)


        model = SSModel(g, dests=goals, betas=[0.2], T=l)
        model.evaluate(test_data)
        model.train(data)
        model.predict(test_data)
        model.evaluate(test_data)

class TestBasicExp(TestCase):
    def test_nocrash(self):
        be = BasicExperiment()
        be.main_loop()
