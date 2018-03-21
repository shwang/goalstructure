import sklearn
import sklearn.metrics
import tensorflow as tf
import numpy as np

from layers import *
import syn

class ExpModel(object):
    def train(self, data):
        """
        Override this method to train the model using the provided dataset.
        """
        raise Exception("not yet implemented")

    def evaluate(self, test_data, training_data=None):
        """
        Evaluate the model by logistic loss and accuracy of prediction on
        test_data.
        """
        if training_data is not None:
            self.train(training_data)
        pred = self.predict(test_data)
        y_true = test_data.Y.reshape(-1)
        if len(pred.shape) == 3:
            y_pred = pred.reshape(-1, pred.shape[2])
        num_labels = y_pred.shape[-1]
        labels = np.arange(num_labels)
        loss = sklearn.metrics.log_loss(y_true=y_true, y_pred=y_pred,
                labels=labels)

        y_pred_best = np.argmax(y_pred, -1)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred_best)
        print loss, acc
        return loss, acc

    def predict(self, X):
        """
        Override this method to return the Model's predictions.
        This method is called by evaluate.
        """
        raise Exception("not yet implemented")

    def close(self):
        pass

class TFModel(ExpModel):
    def __init__(self, tf_model, iters=1000):
        self.tf_model = tf_model
        self.iters = iters
        with self.tf_model.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def train(self, data):
        self.tf_model.train_model(self.sess, data, iters=self.iters)

    def predict(self, data):
        return self.tf_model.predict(self.sess, data)

    def close(self):
        self.sess.close()
        del self.sess
        self.tf_model.close()

from pp.mdp import GridWorldMDP
from pp.inference.hardmax import action
class SSModel(ExpModel):
    def __init__(self, g, dests, betas, T):
        self.g = g
        self.dests = dests
        self.betas = betas
        self.T = T

    def train(self, data):
        # NO-OP
        # TODO: Update destination prior?
        pass

    def predict(self, data):
        trajs, Y = data.trajs, data.Y
        pred = []
        for i, traj in enumerate(trajs):
            res, _, _ = action.infer_joint(g=self.g, dests=self.dests,
                    betas=self.betas, T=self.T, traj=traj)
            pred.append(res)
        pred = np.array(pred)
        return pred


class Experiment(object):
    def generate_datasets(self):
        """
        Called once
        Returns: A list of (name [string], train_dataset, test_dataset) tuples.
        """
        return datas
    def generate_models(self):
        """
        Called several times -- once for each dataset.
        After each dataset is used every model is closed and a fresh model is
        trained for the next dataset.

        Returns: A list of (name [string], model) tuples.
        """
        return models

    def main_loop(self, iters=20):
        res = {}

        for i in range(iters):
            print("Hello, this is iter {}".format(i))
            datasets = self.generate_datasets()
            for dname, train_data, test_data in datasets:
                models = self.generate_models()
                for mname, model in models:
                    key = (dname, mname)
                    if key not in res:
                        res[key] = []

                    print("Training model {} on dataset {}...".format(
                        mname, dname))
                    model.train(train_data)
                    ev = model.evaluate(test_data)
                    print ev
                    res[(dname, mname)].append(ev)
                    model.close()
        return res

class BasicExperiment(Experiment):
    def __init__(self, N=100):
        self.N = N
        self.g = GridWorldMDP(N, N)
        self.goals = [self.g.coor_to_state(N-1, N-1),
                self.g.coor_to_state(0, 0)]
        self.k, self.l = 3, 3

    def generate_datasets(self):
        def helper(samples):
            N = self.N
            g = self.g

            data = syn.gen_predict_actions(self.g, self.goals, self.k, self.l,
                    samples=samples, beta=1e-3)
            test_data = syn.gen_predict_actions(self.g, self.goals, self.k,
                    self.l, samples=10000, beta=1e-3)
            return data, test_data

        res = []
        for s in [128, 500, 1000]:
            data, test_data = helper(s)
            name = "{N}x{N} [{s} samples]".format(N=self.N, s=s)
            res.append([s, data, test_data])
        return res

    def generate_models(self):
        res = []
        # SS
        res.append(["shitscam",
            SSModel(self.g, self.goals, betas=[0.1], T=self.l)])
        # DirectActions
        res.append(["direct act", TFModel(DirectActions(), iters=1000)])
        # LabelEnd
        res.append(["LabelEnd", TFModel(DirectAuxLabelEnd(G=len(self.goals)),
            iters=1000)])
        return res
