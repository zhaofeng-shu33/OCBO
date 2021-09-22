import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

import matplotlib.pyplot as plt

from OCBO.cstrats.profile_cts import ContinuousMultiTaskTS, CMTSPM, ProfileEI
from OCBO.cstrats import copts
from dragonfly.utils.option_handler import load_options

from OCBO.util.misc_util import uniform_draw

def black_box_function_1(vec):
    x, y = vec
    func_val = np.cos(2 * x) * np.cos(y) + np.sin(x)
    constraint_val = np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) + 0.5
    return (-func_val, constraint_val)

np.random.seed(100826730)
function = black_box_function_1 # maximization problem
domain = [[0, 6], [0, 6]]
ctx_dim = 1
max_capital = 10
init_capital = 100

options = load_options(copts)
options.profile_evals = 200
options.num_profiles = 100
options.kernel_type = 'matern'
options.matern_nu = 0.5
model = ProfileEI(function, domain, ctx_dim, options, eval_set=True, is_synthetic=False)
init_pts = list(uniform_draw(domain, init_capital))
# switch off the hyper-parameter tuning of GP
histories = model.optimize(max_capital, init_pts=init_pts, pre_tune=True)
# print(histories)
# plotting the result

ctx_array = np.linspace(0, 6)
n = ctx_array.shape[0]
action = np.zeros(n)
true_action = np.zeros(n)


for i in range(n):
    x = ctx_array[i]
    ctx_plus_action = model._get_ctx_improvement([x], predict=True)
    # extract the action
    action[i] = ctx_plus_action[-1]
    con = lambda y: np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) + 0.5
    nlc = NonlinearConstraint(con, -np.inf, 0.0)
    true_action[i] = minimize(lambda y: np.cos(2 * x) * np.cos(y) + np.sin(x),
        x0=[3.0], bounds=[(0, 6.0)], constraints=nlc).x[0]
    # use scipy to compute the ground truth
    # import pdb
    # pdb.set_trace()
# compute the fitting error
L2_error = np.mean(np.power(true_action - action, 2))
print(L2_error)
plt.plot(ctx_array, action, label='bayesian')
plt.plot(ctx_array, true_action, label='true')
plt.legend()
plt.xlabel('y')
plt.ylabel('z')
plt.savefig('build/cbo_2.png')
plt.show()