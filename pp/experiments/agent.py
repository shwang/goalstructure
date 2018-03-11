class Agent(object):
    def plan(self, ctx, tr_H, state_R):
        raise Exception("not yet implemented")

    def act(self, ctx, tr_H, state_R):
        plan, details = self.plan(ctx, tr_H, state_R)
        if plan is None or len(plan) == 0:
            a = None
        else:
            a = plan[0][1]
        return a, plan, details

    def _plan_with_beta(self, ctx, tr_H, state_R, beta, max_depth=None):
        collide_probs = CollideProbs(ctx=ctx, T=max_depth, traj=traj_H,
                beta=beta)
        return self.plan(ctx, tr_H, state_R, collide_probs)

    def _plan(self, ctx, tr_H, state_R, collide_probs):
        plan, expected_cost = planner_core(ctx=ctx, traj_H=tr_H,
                collide_probs=collide_probs, verbose_return=True)
        details = dict(expected_cost=expected_cost)
        return plan, details

class FixedAgent(object):
    def __init__(self, fixed_beta):
        self.fixed_beta = fixed_beta

    def plan(self, ctx, tr_H, state_R):
        return self._plan_with_beta(ctx, tr_H, state_R, beta=self.fixed_beta)

class MLEAgent(object):
    # Consider setting min and max beta here.
    def __init__(self, min_beta=0.2, max_beta=100, min_iters=15, max_iters=15):
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.min_iters = min_iters
        self.max_iters = max_iters

    def plan(self, ctx, tr_H, state_R):
        g_H = ctx.g_H
        beta = bin_search(g_H, traj_H, g_H.goal, guess=beta_guess,
                verbose=False, k=k, min_beta=self.min_beta,
                max_beta=self.max_beta, min_iters=self.min_iters,
                max_iters=self.max_iters)
        return self._plan_with_beta(ctx, tr_H, state_R, beta=beta)

class BayesAgent(object):
    def __init__(self, betas, priors=None, k=None):
        self.betas = betas
        self.priors = priors

    def plan(self, ctx, tr_H, state_R):
        g_H = ctx.g_H
        collide_probs = CollideProbsBayes(ctx=ctx, betas=betas, priors=priors,
                traj=forget(traj_H, k), T=max_depth)
        P_beta = inf_mod.beta.calc_posterior_over_set(g=g_H, traj=traj_H, k=k,
                goal=g_H.goal, betas=betas, priors=priors)
        plan, ex_cost, final_node =  _robot_planner(ctx, state_R=state_R,
                traj_H=traj_H, collide_probs=collide_probs, verbose_return=True,
                k=k, **kwargs)

    return plan, ex_cost, final_node, P_beta


####################################
# BEGIN CollideProbs...
# XXX: Should this stuff be in its own file?
#####################

class CollideProbs(object):
    def __init__(self, ctx, T=None, traj=[],
            beta=1, start_R=0, state_probs_cached=None, inf_mod=inf_default):
        """
        Helper class for lazily calculating the collide probability at
        various states and timesteps.
        """
        assert len(traj) > 0 or ctx.start_H is not None, \
                "Must set at least one of `traj` or `ctx.start_H`"
        self.ctx = ctx
        self.g = g = ctx.g_H
        self.start_H = ctx.start_H
        self.traj = traj  # Allowed to differ from ctx.traj_H b/c so we can give
                            # simulated Robot a partial trajectory at each given
                            # timestep
        if T is not None:
            self.T = T
        elif ctx.N is not None:
            self.T = ctx.N * 2
        else:
            self.T = ctx.g_R.rows * 2
        self.beta = beta
        self.collide_radius = self.ctx.collide_radius
        self.inf_mod = inf_mod

        if state_probs_cached is not None:
            self.state_probs = state_probs_cached.reshape(T+1, g.rows, g.cols)
        else:
            self.state_probs = self.calc_state_probs()
        self.cache = {}

    def calc_state_probs(self):
        g = self.g
        if len(self.traj) == 0:
            state_probs = self.inf_mod.state.infer_from_start(g, self.start_H,
                    g.goal, T=self.T, beta_or_betas=self.beta)[0].reshape(
                            self.T+1, g.rows, g.cols)
        else:
            state_probs = self.inf_mod.state.infer(g, self.traj,
                    g.goal, T=self.T, beta_or_betas=self.beta)[0].reshape(
                            self.T+1, g.rows, g.cols)
        return state_probs

    def get(self, t, s):
        if (t, s) in self.cache:
            return self.cache[(t, s)]
        x, y = self.g.state_to_coor(s)
        r = self.collide_radius
        colliding = self.state_probs[t, max(x-r,0):x+r+1, max(y-r,0):y+r+1]
        result = np.sum(colliding)
        self.cache[(t, s)] = result
        return result

class CollideProbsBayes(CollideProbs):
    def __init__(self, ctx, betas, priors=None, *args, **kwargs):
        """
        Helper class for lazily calculating the collide probability at
        various states and timesteps.
        """
        self.betas = betas
        self.priors = priors
        CollideProbs.__init__(self, ctx, beta="N/A (Bayes: see self.betas)",
                *args, **kwargs)


    def calc_state_probs(self):
        g = self.g
        state_probs = self.inf_mod.state.infer_bayes(g=g, traj=self.traj,
                init_state=self.start_H, dest=g.goal, T=self.T,
                betas=self.betas, priors=self.priors).reshape(
                        self.T+1, g.rows, g.cols)
        return state_probs
