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
            'expected improvement hyperparameter, which controls the exploitation and exploration trade-off')
]

class ProfileOpt(ContinuousOpt):

    def _child_set_up(self, function, domain, ctx_dim, options):
        self.num_profiles = options.num_profiles
        self.profile_evals = options.profile_evals
        self.xi = options.xi

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
        act_set = sample_grid([list(ctx)], self.act_domain, self.profile_evals)
        means, covmat = self.gp.eval(act_set, include_covar=True)
        best_post = np.min([np.max(means), np.max(self.y_data)])
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

    def _determine_next_query(self):
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
            imp = -res.fun[0]
            if imp > best_imp:
                best_pt = res.x
                best_imp = imp
        # Return the best context and action.
        return best_pt

    def ctx_improvement_func(self, ctx):
        """Get expected improvement over best posterior mean capped by
        the best seen reward so far.
        """
        means, covmat = self.gp.eval(ctx, include_covar=True)
        best_post = np.min([means[0], self.y_max])
        stds = np.sqrt(covmat.diagonal().ravel())
        xi = 0.0
        norm_diff = (means - best_post - xi) / stds
        eis = stds * (norm_diff * normal_distro.cdf(norm_diff) \
                + normal_distro.pdf(norm_diff))
        if self.has_constraint:
            _means, _covmat = self.constraint_gp.eval(ctx, include_covar=True)
            _z = -1.0 * _means / _covmat.diagonal()
            eis *= normal_distro.cdf(_z)
        return eis

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
