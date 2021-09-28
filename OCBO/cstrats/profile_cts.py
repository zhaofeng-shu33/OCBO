"""
Thompson Sampling strategies for continuous context.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro
from scipy.optimize import minimize

from OCBO.cstrats.cts_opt import ContinuousOpt
from dragonfly.utils.option_handler import get_option_specs
from OCBO.util.misc_util import sample_grid, uniform_draw, knowledge_gradient

prof_args = [\
        get_option_specs('num_profiles', False, 50,
            'Number of contexts to consider picking from.'),
        get_option_specs('profile_evals', False, 100,
            'Number of evaluations for each context to determine max.'),
        get_option_specs('xi', False, 0.0,
            'expected improvement hyperparameter, which controls the exploitation and exploration trade-off'),
        get_option_specs('opt_sampling', False, True, 'whether to perform the optimization by finite sampling strategy')            
]

class ProfileOpt(ContinuousOpt):

    def _child_set_up(self, function, domain, ctx_dim, options):
        self.num_profiles = options.num_profiles
        self.profile_evals = options.profile_evals
        self.xi = options.xi
        self.opt_sampling = options.opt_sampling

    def _determine_next_query(self):
        # Get the contexts to test out.
        ctxs = self._get_ctx_candidates(self.num_profiles)
        # For each context...
        best_pt, best_imp = None, float('-inf')
        for ctx in ctxs:
            # Find the best context and give its improvement.
            pt, imp = self._get_ctx_improvement(ctx)
            if imp > best_imp:
                best_pt, best_imp = pt, imp
        # Return the best context and action.
        return best_pt

    def _get_ctx_improvement(self, ctx):
        """Get the improvement for the context.
        Args:
            ctx: ndarray characterizing the context.
        Returns: Best action and the improvement it provides.
        """
        raise NotImplementedError('Abstract Method')

class ProfileEI(ProfileOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'pei'

    def _get_ctx_improvement(self, ctx, predict=False):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        if self.opt_sampling is False:
            return self._get_ctx_improvement_no_sampling(ctx, predict=predict)
        #if predict:
        #    _, act = self.get_maximal_mean(ctx)
        #    return np.hstack((ctx, act))

        act_set = sample_grid([list(ctx)], self.act_domain, self.profile_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.min([np.max(means), self.y_max])
        stds = np.sqrt(covmat.diagonal().ravel())
        if predict:
            xi = 0.0
        else:
            xi = self.xi
        norm_diff = (means - best_post - xi) / stds
        eis = stds * (norm_diff * normal_distro.cdf(norm_diff) \
                + normal_distro.pdf(norm_diff))
        if self.has_constraint:
            _means, _covmat = self.constraint_gp.eval(act_set, include_covar=True)
            _z = -1.0 * _means / _covmat.diagonal()
            eis *= normal_distro.cdf(_z)
        ei_val = np.max(eis)
        ei_pt = act_set[np.argmax(eis)]
        if predict:
            return ei_pt
        return ei_pt, ei_val

    def _get_ctx_improvement_no_sampling(self, ctx, predict=False):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far. No sampling, optimize using some gradient methods
        """
        # obtain the best act by solving a non-linear equation
        def negative_PEI_star(act, best_post):
            # concantenate ctx with act to obtain the whole vector
            act_set = np.hstack((ctx, act)).reshape(1, -1)
            means, covmat = self.gp.eval(act_set, include_covar=True)        
            stds = np.sqrt(covmat.diagonal().ravel())
            if predict:
                xi = 0.0
            else:
                xi = self.xi
            norm_diff = (means - best_post - xi) / stds
            eis = stds * (norm_diff * normal_distro.cdf(norm_diff) \
                    + normal_distro.pdf(norm_diff))
            if self.has_constraint:
                _means, _covmat = self.constraint_gp.eval(act_set, include_covar=True)
                _z = -1.0 * _means / _covmat.diagonal()
                eis *= normal_distro.cdf(_z)
            return -1.0 * eis[0]
        best_imp = -np.infty
        ei_pt = None
        for _ in range(self.options.profile_evals):
            max_mean, candidate_act = self.get_maximal_mean(ctx)
            # best_post = np.min([max_mean, self.y_max]) # T_alpha
            # minimize the function negative_PEI_star
            # res = minimize(lambda x: negative_PEI_star(x, best_post),
            #                uniform_draw(self.act_domain, 1).reshape(1),
            #                bounds=self.act_domain,
            #                method="L-BFGS-B")
            # if not res.success:
            #    import pdb
            #    pdb.set_trace()
            if max_mean > best_imp:
                best_imp = max_mean
                ei_pt = np.hstack((ctx, candidate_act))
        if predict:
            return ei_pt
        ei_val = best_imp
        return ei_pt, ei_val

    def _determine_next_query(self):
        if self.opt_sampling:
            return super(ProfileEI, self)._determine_next_query()
        # Get the contexts to test out.
        ctxs = uniform_draw(self.domain, self.num_profiles)
        # For each context...
        # Explore the parameter space more thoroughly
        best_pt, best_imp = None, float('-inf')
        for ctx in ctxs:
            # Find the best context and give its improvement.
            res = minimize(lambda x: -1.0 * self.ctx_improvement_func(x.reshape(1, -1)),
                           ctx.reshape(1, -1),
                           bounds=self.domain,
                           method="L-BFGS-B")
            imp = -res.fun
            if imp > best_imp:
                best_pt = res.x
                best_imp = imp
        # Return the best context and action.
        return best_pt

    def get_maximal_mean(self, task, init_action=None):
        if init_action is None:
            action = uniform_draw(self.act_domain, 1)
        else:
            action = init_action
        res = minimize(lambda x: -1.0 * self.gp.eval(np.hstack((task, x)).reshape(1, -1))[0],
                           action,
                           bounds=self.act_domain,
                           method="L-BFGS-B")
        max_mean = -res.fun
        return max_mean, res.x

    def ctx_improvement_func(self, ctx):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        # extract task part from ctx
        task = ctx[0, :self.ctx_dim]
        action = ctx[0, self.ctx_dim:]
        # embedded optimization
        max_mean, _ = self.get_maximal_mean(task, init_action=action)
        best_post = np.min([max_mean, self.y_max])
        means, covmat = self.gp.eval(ctx, include_covar=True)
        stds = np.sqrt(covmat.diagonal().ravel())
        xi = 0.0
        norm_diff = (means - best_post - xi) / stds
        eis = stds * (norm_diff * normal_distro.cdf(norm_diff) \
                + normal_distro.pdf(norm_diff))
        if self.has_constraint:
            _means, _covmat = self.constraint_gp.eval(ctx, include_covar=True)
            _z = -1.0 * _means / _covmat.diagonal()
            eis *= normal_distro.cdf(_z)
        return eis[0]

class CMTSPM(ProfileOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'cmts-pm'

    def _get_ctx_improvement(self, ctx, predict=False):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        act_set = sample_grid([list(ctx)], self.act_domain, self.profile_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.min([np.max(means), np.max(self.y_data)])
        if predict:
            return act_set[np.argmax(means)]
        sample = self.gp.draw_sample(means=means, covar=covmat).ravel()
        gain = np.max(sample) - best_post
        best_pt = act_set[np.argmax(sample)]
        return best_pt, gain

class ContinuousMultiTaskTS(ProfileOpt):

    @staticmethod
    def get_strat_name():
        """Get the name of the strategies."""
        return 'cmts'

    def _get_ctx_improvement(self, ctx, predict=False):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        act_set = sample_grid([list(ctx)], self.act_domain, self.profile_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.argmax(means)
        if predict:
            return act_set[best_post]
        sample = self.gp.draw_sample(means=means, covar=covmat).ravel()
        gain = np.max(sample) - sample[best_post]
        best_pt = act_set[np.argmax(sample)]
        return best_pt, gain

prof_strats = [Namespace(impl=ProfileEI, name=ProfileEI.get_strat_name()),
               Namespace(impl=CMTSPM, name=CMTSPM.get_strat_name()),
               Namespace(impl=ContinuousMultiTaskTS,
                         name=ContinuousMultiTaskTS.get_strat_name()),
]
