from __future__ import division

import numpy as np
import destination
import beta as bt

from ...mdp import GridWorldExpanded

from ...parameters import val_default
from ...util.args import unpack_opt_list

import state
def infer_joint(g, dests, betas, T, *args, **kwargs):
    """
    Calculate the expected action probabilities at each time step by taking a
    linear combination over the state probabilities associated with each
    dest-beta pair.
    Params:
        Same as those of hardmax.state.infer_joint.
    Returns:
        P_res [np.ndarray]: A (T x A) array, where the `t`th entry is the
            contains the probabilities of each action in `t+1` timesteps from
            now.
        P_all [np.ndarray]: A (|dests| x |betas| x T x A) array, where the
            `(d, b)`th entry contains
            the (T x A) expected action probabilities if it were the case
            that `dest == dests[d] and beta_star == beta[b]`.
        P_joint_DB [np.ndarray]: A (|dests| x |betas|) dimension array, where
            the `b`th entry is the posterior probability associated with
            `betas[b]`.
    """
    kwargs["verbose_return"] = True
    _, occ_all, P_joint_DB = state.infer_joint(g, dests, betas, T, *args,
            **kwargs)

    # We want to predict actions for t=(1, 2, .., T+1). Truncate the data so
    # that we avoid calculating the `T+2`th action.
    occ_all = occ_all[:, :, :-1, :]

    n_D, n_B = len(dests), len(betas)
    assert P_joint_DB.shape == (n_D, n_B)
    assert occ_all.shape == (n_D, n_B, T, g.S)

    P_joint_DBtS = np.multiply(P_joint_DB.reshape(n_D, n_B, 1, 1), occ_all)
    P_all = np.empty([n_D, n_B, T, g.A])
    for i, dest in enumerate(dests):
        for j, beta in enumerate(betas):
            P_act = g.action_probabilities(goal=dest, beta=beta)
            P_all[i, j] = np.matmul(P_joint_DBtS[i, j], P_act)

    P_res = np.sum(np.sum(P_all, axis=0), axis=0)
    assert P_res.shape == (T, g.A)

    return P_res, P_all, P_joint_DB
