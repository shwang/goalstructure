import numpy as np
import numpy.random as rand

from pp.mdp import GridWorldMDP
from pp.util.hardmax.simulate import simulate, sample_action
from pp.util.util import display

def rand_traj(g, goal, path_length, beta=1e-4, max_attempts=10):
    res = None
    for i in range(max_attempts):
        s = rand.randint(g.S)
        init_x, init_y = g.state_to_coor(s)
        # TODO: change simulate to have a return None option if trajectory
        # doesn't meet required path_length..
        traj = simulate(g, s, goal, beta=beta, path_length=path_length)
        if path_length == None:
            res = traj
            break
        if len(traj) >= path_length:
            traj = traj[:path_length]
            res = traj
            break
        elif len(traj) < path_length:
            continue
    return res

class Data(object):
    def __init__(self, X, Y, Z=None, name="default", trajs=None, meta=None):
        assert len(X) == len(Y)
        self.N = len(X)
        self.X = np.array(X)
        self.Y = np.array(Y)
        if Z is None:
            self.Z = Z
        else:
            self.Z = np.array(Z)
        self.trajs = trajs
        self.name = name
        if meta is None:
            meta = {}
        self.meta = meta

    def get_batch(self, size, enable_aux=False):
        """
        Samples X, Y from the data without replacement.

        Parameters:
        size -- The number of samples to return.
        enable_aux [boolean] -- (optional) If False, only return X and Y. If
            True, then also sample auxiliary outputs Z and return thoses.

        Returns:
        X -- A rank 2 tensor of dimensions [size x input_size].
        Y -- A rank 2 tensor of dimensions [size x output_size].
        Z -- A rank 2 tensor of dimensions [size x aux_size]. (only returned if
            enable_aux is True.
        """

        assert 0 <= size <= self.N, size
        indices = rand.choice(range(self.N), size=size, replace=False)
        if not enable_aux:
            return self.X[indices], self.Y[indices]
        else:
            return self.X[indices], self.Y[indices], self.Z[indices]


def gen_predict_policy(g, goals, samples=10, beta=1e-5):
    """
    Generate a dataset of state-action pairs for policy prediction. Each
    data point is generated as follows:
    1. Uniformly choose a goal from the list of goals.
    2. Uniformly choose a state.
    3. Beta-rationally sample an action (using the Q function associated with
    the goal).

    Returns:
    data -- A Data object with the following parameters:
        X -- [np.ndarray] 2-length vectors describing the x and y coordinates
            of the states from which actions are sampled, in an array of
            dimensions [`samples`, 2].
        Y -- [np.ndarray] The index of the next action. If Y[i] = a, then the
            action associated with the ith sample is g.Actions[i]. Y is an array
            of length `samples`.
        Z -- [np.ndarray] The index of the goal. Z is an array of length `samples`.
        [`samples` x `len(goals)`].
    """
    X = np.empty([samples, 2], dtype=float)
    Y = np.empty([samples], dtype=float)
    Z = np.empty([samples], dtype=float)

    for i in range(samples):
        goal_ind = rand.randint(len(goals))
        goal = goals[goal_ind]
        s = rand.randint(g.S)
        coor = g.state_to_coor(s)
        action_ind = sample_action(g, s, goal, beta=beta)

        X[i] = coor
        Y[i] = action_ind
        Z[i] = goal_ind

    return Data(X, Y, Z)

def gen_predict_policy2(g, goals, samples=10, beta=1e-5):
    """
    Generate a dataset of (initial_state, state, action) triples for policy
    prediction. Each data point is generated as follows:
    1. Uniformly choose a goal from the list of goals.
    2. Uniformly choose an initial_state and generate a beta-rationally
    sampled trajectory to the goal.
    3. Uniformly choose some timestep of the sample trajectory, from which we
    pull the (state, action) pair.

    Returns:
    data -- A Data object with the following parameters:
        X -- [np.ndarray] vectors of length 4, where the first 2 elements
            indicate the x and y coordinates of the initial state and the
            second 2 elements indicate the x and y coordinates of the current
            state. X is an array of dimensions (`samples` x 4).
        Y -- [np.ndarray] Integer labels, in an array of length `samples. Each
            label corresponds to the index in `g.Actions` of the next action.
        Z -- [np.ndarray] Integer labels, in an array of length `samples`. Each
            label corresponds to the index in `goals` of the trajectory's goal.
    """
    X = np.empty([samples, 4], dtype=float)
    Y = np.empty([samples], dtype="uint32")
    Z = np.empty([samples], dtype="uint32")

    for i in range(samples):
        goal_ind = rand.randint(len(goals))
        goal = goals[goal_ind]
        traj = rand_traj(g, goal, None, beta=beta)

        init_s = traj[0][0]
        s, a = traj[rand.randint(len(traj))]

        X[i][0:2] = g.state_to_coor(init_s)
        X[i][2:4] = g.state_to_coor(s)
        Y[i] = a
        Z[i] = goal_ind

    return Data(X, Y, Z)


def gen_predict_goal(g, goals, k, samples=10, beta=1e-3):
    """
    Generate a dataset for goal prediction. Each trajectory is generated as
    follows:
    1. Uniformly choose a goal from the list of goals
    2. Uniformly choose a starting location and generate a beta-rationally
    sampled trajectory to the goal.
        a) If this trajectory has length greater than k, then truncate the
        trajectory to length k.
        b) If this trajectory has length less than k, then repeat step 2 with
        a new starting location.

    Returns:
    data -- A Data object where X and Y are:
        X -- [np.ndarray] k-length trajectories, in an array of dimensions
            (`samples` x `2*k`). The `2*t`th and `2*t+1`th entries of each
            row correspond to the x and y coordinates of the trajectory at time
            `t`.
        Y -- [np.ndarray] Integer labels, in an array of length `samples`. Each
            label corresponds to the index in `goals` of the trajectory's goal.
    """
    X = np.empty([samples, 2*k], dtype=np.uint)
    Y = np.empty([samples], dtype=np.uint)
    assert len(goals) > 0

    for i in range(samples):
        y = rand.randint(len(goals))
        goal = goals[y]
        while True:
            s = rand.randint(g.S)
            init_x, init_y = g.state_to_coor(s)
            traj = simulate(g, s, goal, beta=beta, path_length=k)
            if len(traj) > k:
                traj = traj[:k]
            elif len(traj) < k:
                continue
            break

        x = X[i]
        for t, (s, a) in enumerate(traj):
            x[2*t], x[2*t + 1] = g.state_to_coor(s)
        Y[i] = y
    return Data(X, Y)

def gen_predict_traj(g, goals, k, l, samples=10, beta=1e-3):
    """
    Generate a dataset for trajectory prediction. Each trajectory is generated
    as follows:
    1. Uniformly choose a goal from the list of goals
    2. Uniformly choose a starting location and generate a beta-rationally
    sampled trajectory to the goal.
        a) If this trajectory has length greater than (k +l), then truncate the
        trajectory to length (k + l).
        b) If this trajectory has length less than k, then repeat step 2 with
        a new starting location.

    Returns:
    data -- A Data object where X and Y are:
        X -- [np.ndarray] k-length trajectories, in an array of dimensions
            (`samples` x `2*k`). The `2*t`th and `2*t+1`th entries of each
            row correspond to the x and y coordinates of the trajectory at time
            `t`.
        Y -- [np.ndarray] l-length trajectories, in an array of dimensions
            (`samples` x `2*l`). The `2*t`th and `2*t+1`th entries of each
            row correspond to the x and y coordinates of the trajectory at time
            `k + t`.
        Z -- [np.ndarray] Integer labels, in an array of length `samples`. Each
            label corresponds to the index in `goals` of the trajectory's goal.
    """
    X = np.empty([samples, 2*k], dtype=np.uint)
    Y = np.empty([samples, 2*l], dtype=np.uint)
    Z = np.empty([samples], dtype=np.uint)
    assert len(goals) > 0
    for i in range(samples):
        goal_ind = rand.randint(len(goals))
        goal = goals[goal_ind]
        while True:
            s = rand.randint(g.S)
            init_x, init_y = g.state_to_coor(s)
            traj = simulate(g, s, goal, beta=beta, path_length=k+l)
            if len(traj) > k+l:
                traj = traj[:k+l]
            elif len(traj) < k+l:
                continue
            break

        x = X[i]
        y = Y[i]
        for t, (s, a) in enumerate(traj):
            if t < k:
                x[2*t], x[2*t + 1] = g.state_to_coor(s)
            else:
                y[2*(t-k)], y[2*(t-k) + 1] = g.state_to_coor(s)
        Z[i] = goal_ind
    return Data(X, Y, Z)

import random
def puddles_world(N=100, p=0.2, puddle_reward=-2):
    """
    Generate a world where some squares have scaled reward -2 or -2*sqrt(2).
    """
    reward_dict = {}
    for x in range(N):
        for y in range(N):
            if random.random() < p:
                reward_dict[(x, y)] = -2

    g = GridWorldMDP(N, N, reward_dict=reward_dict)
    return g

def gen_predict_actions(g, goals, k, l, samples=10, beta=1e-3):
    """
    Generate a dataset for trajectory to action prediction. Each trajectory is
    generated as follows:
    1. Uniformly choose a goal from the list of goals
    2. Uniformly choose a starting location and generate a beta-rationally
    sampled trajectory to the goal.
        a) If this trajectory has length greater than (k +l), then truncate the
        trajectory to length (k + l).
        b) If this trajectory has length less than k, then repeat step 2 with
        a new starting location.

    Returns:
    data -- A Data object where X and Y are:
        X -- [np.ndarray] k-length trajectories, in an array of dimensions
            (`samples` x `2*k`). The `2*t`th and `2*t+1`th entries of each
            row correspond to the x and y coordinates of the trajectory at time
            `t`.
        Y -- [np.ndarray] l-length action sequences, in an array of dimensions
            (`samples` x `l`). The `t`th entry of each
            row corresponds to the index (w.r.t. g.Actions) of the action at
            time `k + t`.
        Z -- [np.ndarray] Integer labels, in an array of length `samples`. Each
            label corresponds to the index in `goals` of the trajectory's goal.
    """

    X = np.empty([samples, 2*k], dtype=np.uint)
    Y = np.empty([samples, l], dtype=np.uint)
    Z = np.empty([samples], dtype=np.uint)
    assert len(goals) > 0
    trajs = []
    for i in range(samples):
        goal_ind = rand.randint(len(goals))
        goal = goals[goal_ind]
        traj = rand_traj(g, goal, path_length=k+l, beta=beta)
        if traj == None:
            raise Exception("Failed to produce trajectory")
        trajs.append(traj)

        x = X[i]
        y = Y[i]
        for t, (s, a) in enumerate(traj):
            if t < k:
                x[2*t], x[2*t + 1] = g.state_to_coor(s)
            else:
                y[t-k] = a
        Z[i] = goal_ind
    return Data(X, Y, Z, trajs=trajs)
