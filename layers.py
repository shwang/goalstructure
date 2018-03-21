import tensorflow as tf
from tensorflow import nn
import numpy as np
from pp.mdp import GridWorldMDP

import syn

class NN(object):
    """
    A neural net with the following properties.
    Input: `input_size` nodes.
    Output: `output_size` nodes.
    1 Hidden layer, with relu activations.
    """

    def __init__(self, input_size, output_size, hidden_size=32,
            use_y_sparse=False, num_labels=1, y_hat_shape=None,
            aux_size=2):
        """
        K is the number of input steps.
        L is the number of output steps.
        hidden_size is the number of nodes in each of the two hidden layers.

        y_hat_shape is a Shape tuple used to reshape the neural net's output
            before the output is compared against the labels.
            `self.Y_hat` is defined as `tf.reshape(self.out, y_hat_shape)` if
            this parameter is provided. Otherwise `self.Y_hat = self.out`.

        use_y_sparse [bool] -- This is an ugly way to make the trainer use the
            self.Y_sparse placeholder. There is probably a cleaner way to do
            this.
        num_labels [int] -- Only relevant when use_y_sparse is True. Should be
            set to the number of classifications made by the neural network.
            This number determines the shape of self.Y_sparse.
            Again, the cleanness of this parameter is less than ideal.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.y_hat_shape = y_hat_shape
        self.use_y_sparse = use_y_sparse
        self.num_labels = num_labels
        self.graph = tf.Graph()
        self.init_training()

    def close(self):
        del self.graph
        print("closing the graph :~")

    def xavier_init(self, size):
        with self.graph.as_default():
            in_dim = size[0]
            xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
            return tf.random_normal(shape=size, stddev=xavier_stddev)

    def init_training(self):
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, self.input_size],
                    name="X")
            self.Y = tf.placeholder(tf.float32, shape=[None, self.output_size],
                    name="Y")
            if self.num_labels == 1:
                self.Y_sparse = tf.placeholder(tf.int32, shape=[None],
                        name="Y_sparse")
            else:
                self.Y_sparse = tf.placeholder(tf.int32,
                        shape=[None, self.num_labels],
                        name="Y_sparse")

            H = self.hidden_size
            self.W1 = tf.Variable(self.xavier_init([self.input_size, H]),
                    name="W1")
            self.b1 = tf.Variable(tf.zeros(shape=[H]), name="b1")

            self.W2 = tf.Variable(self.xavier_init([H, H]), name="W2")
            self.b2 = tf.Variable(tf.zeros(shape=[H]))

            self.W3 = tf.Variable(self.xavier_init([H, self.output_size]))
            self.b3 = tf.Variable(tf.zeros(shape=[self.output_size]))

            self.theta = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]

            # Connect layers
            self.ho1 = tf.matmul(self.X, self.W1) + self.b1
            self.ho1a = tf.nn.leaky_relu(self.ho1)
            self.ho2 = tf.matmul(self.ho1a, self.W2) + self.b2
            self.ho2a = tf.nn.leaky_relu(self.ho2)
            self.ho3 = tf.matmul(self.ho2a, self.W3) + self.b3
            self.out = self.ho3a = tf.nn.leaky_relu(self.ho3)

            if self.y_hat_shape is not None:
                self.Y_hat = tf.reshape(self.out, self.y_hat_shape)
            else:
                self.Y_hat = self.out

            self.build_loss()
            self.build_solver()

    def build_loss(self):
        """
        Override this method to implement custom loss, stored as self.loss.
        It is okay if self.loss is a vector, because it will be averaged before
        back propagation.

        By default the loss is either the l2 norm (if use_y_sparse==False)
            or the sparse softmax cross entropy (if use_y_sparse==True).
        """
        with self.graph.as_default():
            if not self.use_y_sparse:
                self.loss = tf.norm(self.Y_hat - self.Y, axis=0)
            else:
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.Y_sparse, logits=self.Y_hat)


    def build_solver(self):
        """
        Override this class to implement a custom solver functions,
        which are stored as self.solver.

        By default we use a default AdamOptimizer over self.theta, minimizing
        `self.loss`.
        """
        with self.graph.as_default():
            self.solver = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(
                    self.loss, var_list=self.theta)

    def train_model(self, sess, data, mb_size=128, iters=1000, test_data=None):
        """
        Train the model.

        Parameters:
        sess [tf.Session] -- trivial.
        data [syn.Data] -- Training data.
        mb_size [int] -- Minibatch size.
        """
        with self.graph.as_default():
            for it in range(iters):
                X, Y = data.get_batch(mb_size)
                feed_dict = self.data_to_feed_dict(data, mb_size)
                avg_loss, _ = sess.run([tf.reduce_mean(self.loss), self.solver],
                        feed_dict=feed_dict)

                if it % 100 == 0:
                    print("iter {}:".format(it))
                    print("avg_loss: {:.4}".format(avg_loss))
                    if test_data is not None:
                        self.assess_model(sess, test_data)
                    print("")

    def data_to_feed_dict(self, data, mb_size):
        with self.graph.as_default():
            X, Y = data.get_batch(mb_size)
            if self.use_y_sparse:
                Y_ph = self.Y_sparse
            else:
                Y_ph = self.Y
            return {self.X: X, Y_ph: Y}

    def predict(self, sess, data):
        with self.graph.as_default():
            pred = tf.nn.softmax(self.Y_hat, axis=-1)
        return sess.run(pred, feed_dict={self.X: data.X, self.Y_sparse: data.Y})

    def predict_best(self, sess, data):
        """
        Return the index of the best, rather than the probability of each
        label.
        """
        with self.graph.as_default():
            pred = tf.nn.softmax(self.Y_hat, axis=-1)
        return sess.run(tf.argmax(pred, -1),
                feed_dict={self.X: data.X, self.Y_sparse: data.Y})


    def assess_model(self, sess, data):
        with self.graph.as_default():
            X, Y = data.X, data.Y
            if self.use_y_sparse:
                pred = tf.nn.softmax(self.Y_hat, axis=-1)
                correct_prediction = tf.equal(tf.argmax(pred, -1), Y)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                acc = accuracy.eval({self.X: X, self.Y_sparse: Y}, session=sess)
                print("Accuracy:", acc)

            # TODO: create debugging/assessment interpretation tools
            # if acc > .85:
            #     p, correct_pred = sess.run([pred, correct_prediction],
            #             feed_dict={self.X: X, self.Y_sparse: Y})
            #     for i, flag in enumerate(correct_pred):
            #         if not flag:
            #             x = X[i]
            #             y = GridWorldMDP.Actions(Y[i])
            #             x_hat = X[i]
            #             y_hat = GridWorldMDP.Actions(np.argmax(p[i]))
            #             if y == y_hat:
            #                 import pdb; pdb.set_trace()
            #             p_y = p[i][y]
            #             print ("Desired: {x} => {y}. Got: {x_hat} => {y_hat}."
            #                 + " P({y})={p_y:.3f}").format(x=x, y=str(y),
            #                         x_hat=x_hat, y_hat=str(y_hat), p_y=p_y)


class DirectAction(NN):
    """
    A neural net with the following properties.
    Input: 2 nodes, corresponding to the x and y coordinates of the state.
    Output: `A` logit nodes, one for each action in the GridWorld.
    """

    def __init__(self, A=9, hidden_size=128):
        NN.__init__(self, input_size=2, output_size=A, use_y_sparse=True)


class DirectActions(NN):
    """
    A neural net with the following properties.
    Input: 2*k nodes. The `2*t`th and `2*t+1`th inputs
        correspond to the x and y coordinates of the trajectory at time `t`.
    Output: A*l nodes. The (A*t, A*t+1, ..., A*t+L-1) outputs correspond to the
        pre-softmax logits of the action at time `k+t+1`.
    """

    def __init__(self, k=3, l=3, A=9, **kwargs):
        """
        K is the number of input steps.  L is the number of output steps.
        """
        self.A = A
        self.k = k
        self.l = l
        NN.__init__(self, input_size=k*2, output_size=A*l, use_y_sparse=True,
                num_labels=l, y_hat_shape=(-1, self.l, self.A), **kwargs)

    def build_loss(self):
        with self.graph.as_default():
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.Y_sparse,
                    logits=self.Y_hat)

    def experiment(self, N=10, iters=1000, mb_size=128, samples=10000):
        g = GridWorldMDP(N, N)
        goals = [g.coor_to_state(N-1, N-1)]

        data = syn.gen_predict_actions(g, goals, self.k, self.l,
                samples=samples, beta=1e-3)
        test_data = syn.gen_predict_actions(g, goals, self.k, self.l,
                samples=100, beta=1e-3)


        with self.graph.as_default():
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            self.train_model(sess, data, mb_size=mb_size, iters=iters,
                    test_data=test_data)
            self.assess_model(sess, test_data)

class NNAux(NN):
    """
    Abstract class for NN that also predicts a goal as auxiliary output.
    """

    def __init__(self, G, k=3, l=3, A=9, aux_loss_weight=0.5,
            use_z_sparse=True, **kwargs):
        """
        k is the number of input steps.  l is the number of output steps.

        aux_loss_weight is a number between 0 and 1 indicating the proportion of
        the training loss determined by auxiliary goal prediction loss (as
        opposed to action prediction loss).
        """
        self.A = A
        self.G = G
        self.k = k
        self.l = l
        self.aux_loss_weight = aux_loss_weight
        self.use_z_sparse = use_z_sparse  # This will be moved to the superclass?
        self.aux_ready = False
        NN.__init__(self, input_size=k*2, output_size=A*l, use_y_sparse=True,
                num_labels=l, y_hat_shape=(-1, self.l, self.A), **kwargs)

    def assess_model(self, sess, data):
        with self.graph.as_default():
            NN.assess_model(self, sess, data)
            X, Z = data.X, data.Z

            pred = tf.nn.softmax(self.aux, axis=-1)
            correct_prediction = tf.equal(tf.argmax(pred, -1), Z)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            acc = accuracy.eval({self.X: X, self.Z_sparse: Z}, session=sess)
            print("Aux Accuracy:", acc)

    def build_loss(self):
        if not self.aux_ready:
            return
        with self.graph.as_default():
            NN.build_loss(self)
            self.main_loss = self.loss
            if self.use_z_sparse:
                Z_ph = self.Z_sparse
            else:
                Z_ph = self.Z
            self.aux_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=Z_ph,
                    logits=self.aux,
                    )
            w = self.aux_loss_weight
            self.loss = tf.reduce_mean(self.aux_loss * w) + \
                    tf.reduce_mean(self.main_loss * (1-w))

    def init_training(self):
        """
        Override this function. Call it, and then call self.build_loss() and
        self.build_solver(). For an example see DirectAuxLabel.init_training().
        """
        NN.init_training(self)

        with self.graph.as_default():
            # True (x, y) coordinates of goal.
            self.Z = tf.placeholder(tf.float32, shape=[None, 2], name="Z")
            # True goal label.
            self.Z_sparse = tf.placeholder(tf.int32, shape=[None],
                    name="Z_sparse")

            self.aux_ready = True   # XXX: Might be cleaaner to have a
                                    # NN.init_training option that skips
                                    # self.build_loss() instead.

    def build_solver(self):
        if not self.aux_ready:
            return
        NN.build_solver(self)

    def data_to_feed_dict(self, data, mb_size):
        X, Y, Z = data.get_batch(mb_size, enable_aux=True)
        Y_ph = self.Y_sparse if self.use_y_sparse else self.Y
        Z_ph = self.Z_sparse if self.use_z_sparse else self.Z
        return {self.X: X, Y_ph: Y, Z_ph: Z}

    def experiment(self, N=10, iters=1000, mb_size=128, samples=10000):
        g = GridWorldMDP(N, N)
        goals = [g.coor_to_state(N-1, N-1), g.coor_to_state(0, 0)]
        assert self.G == len(goals)

        data = syn.gen_predict_actions(g, goals, self.k, self.l,
                samples=samples, beta=1e-3)
        test_data = syn.gen_predict_actions(g, goals, self.k, self.l,
                samples=100, beta=1e-3)

        with self.graph.as_default():
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            self.train_model(sess, data, mb_size=mb_size, iters=iters,
                    test_data=test_data)
            self.assess_model(sess, test_data)

class DirectAuxLabelEnd(NNAux):
    """
    A neural net with the following properties.
    Input: 2 nodes, corresponding to the x and y coordinates of the state.
    Output: `l*A` logit nodes, one for each action at timesteps 1, 2, ..., l.
            Followed by `G` logit nodes, one for each possible goal in the
            dataset.
    """

    def init_training(self):
        with self.graph.as_default():
            NNAux.init_training(self)
            H = self.hidden_size

            self.W_aux = tf.Variable(self.xavier_init([H, self.G]),
                    name="W_aux")
            self.oaux = tf.matmul(self.ho2a, self.W_aux)
            self.aux = self.oauxa = tf.nn.leaky_relu(self.oaux, name="aux")
            self.theta += [self.W_aux]

            self.build_loss()
            self.build_solver()

class DirectAuxLabelMid(NNAux):
    """
    A neural net with the following properties.
    Input: 2 nodes, corresponding to the x and y coordinates of the state.
    Output: `l*A` logit nodes, one for each action at timesteps 1, 2, ..., l.

    The second hidden layer is augmented with G logit nodes.
    One for each possible goal in the dataset.
    """
    def init_training(self):
        """
        I needed to change the dimensions of the second hidden layer, so might
        as well just rewrite this from the beginning. Blah.
        I guess I could abstract away some of these things later...
        """

        # Actually, I'm not going to change those dimensions. Instead,
        # I'm going to make `self.G` of the nodes into aux nodes.
        # TODO: Remove the repetition by calling super.init_training

        # XXX: Move placeholder definitions into its own helper function in
        # class NN.
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, self.input_size],
                    name="X")
            self.Y = tf.placeholder(tf.float32, shape=[None, self.output_size],
                    name="Y")
            if self.num_labels == 1:
                self.Y_sparse = tf.placeholder(tf.int32, shape=[None],
                        name="Y_sparse")
            else:
                self.Y_sparse = tf.placeholder(tf.int32,
                        shape=[None, self.num_labels],
                        name="Y_sparse")

            # True (x, y) coordinates of goal.
            self.Z = tf.placeholder(tf.float32, shape=[None, 2], name="Z")
            # True goal label.
            self.Z_sparse = tf.placeholder(tf.int32, shape=[None], name="Z_sparse")

            H = self.hidden_size
            self.W1 = tf.Variable(self.xavier_init([self.input_size, H]), name="W1")
            self.b1 = tf.Variable(tf.zeros(shape=[H]), name="b1")

            self.W2 = tf.Variable(self.xavier_init([H, H]), name="W2")
            self.b2 = tf.Variable(tf.zeros(shape=[H]))

            self.W3 = tf.Variable(self.xavier_init([H, self.output_size]))
            self.b3 = tf.Variable(tf.zeros(shape=[self.output_size]))

            self.theta = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]

            # Connect layers
            self.ho1 = tf.matmul(self.X, self.W1) + self.b1
            self.ho1a = tf.nn.leaky_relu(self.ho1)
            self.ho2 = tf.matmul(self.ho1a, self.W2) + self.b2
            self.ho2a = tf.nn.leaky_relu(self.ho2)  # Aux calculations are here!
            self.ho3 = tf.matmul(self.ho2a, self.W3) + self.b3
            self.out = self.ho3a = tf.nn.leaky_relu(self.ho3)

            self.aux = self.ho2a[:, -self.G:]

            if self.y_hat_shape is not None:
                self.Y_hat = tf.reshape(self.out, self.y_hat_shape)
            else:
                self.Y_hat = self.out

            self.aux_ready = True   # XXX: Might be cleaaner to have a
                                    # NN.init_training
                                    # option that skips self.build_loss() instead.

            self.build_loss()
            self.build_solver()
