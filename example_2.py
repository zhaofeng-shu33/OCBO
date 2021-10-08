# python example_2.py --opt_sampling
import argparse
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
    return (func_val, constraint_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_profiles', type=int, default=100, help='number of samples in context space')
    parser.add_argument('--train_size', type=int, default=10)
    # parser.add_argument('--profile_evals', type=int, default=200, help='number of samples in action space')
    parser.add_argument('--init_capital', type=int, default=100)
    parser.add_argument('--xi', type=float, default=0.0)
    parser.add_argument('--kernel_type', choices=['rbf', 'matern', 'rq'], default='matern')
    parser.add_argument('--n_restarts_optimizer', type=int, default=1)
    parser.add_argument('--matern_nu', type=float, default=1.5)
    parser.add_argument('--opt_sampling', default=False, const=True, nargs='?')
    parser.add_argument('--profile_evals', type=int, default=200)
    parser.add_argument('--num_profiles', type=int, default=100)
    parser.add_argument('--gp_engine', choices=['dragonfly', 'sklearn'], default='dragonfly')

    args = parser.parse_args()


    np.random.seed(100826730)
    function = black_box_function_1 # maximization problem

    domain = [[0, 6], [0, 6]]
    ctx_dim = 1
    max_capital = args.train_size
    init_capital = args.init_capital

    options = load_options(copts)
    options.opt_sampling = args.opt_sampling
    if options.opt_sampling:
        options.profile_evals = args.profile_evals
        options.num_profiles = args.num_profiles
    else:
        options.profile_evals = 20
        options.num_profiles = 10        
    options.xi = args.xi
    options.gp_engine = args.gp_engine
    options.kernel_type = args.kernel_type
    options.matern_nu = args.matern_nu
    options.hp_samples = args.n_restarts_optimizer
    model = ProfileEI(function, domain, ctx_dim, options, eval_set=True, is_synthetic=False)
    init_pts = list(uniform_draw(domain, init_capital))
    # switch off the hyper-parameter tuning of GP
    histories = model.optimize(max_capital, init_pts=init_pts, pre_tune=True)
    # print(histories)
    # plotting the result

    # inspect hyperparameters of the GP
    hps = model.gp.get_kernel_hps()
    print(hps)

    ctx_array = np.linspace(0, 6)
    n = ctx_array.shape[0]
    pred_max = np.zeros(n)
    true_max_val = np.zeros(n)


    for i in range(n):
        x = ctx_array[i]
        ctx_plus_action = model._get_ctx_improvement([x], predict=True)
        # extract the action
        pred_max[i] = np.cos(2 * x) * np.cos(ctx_plus_action[-1]) + np.sin(x)
        y_samples = np.linspace(0, 6, 500)
        y_samples_index = np.where(np.cos(x) * np.cos(y_samples) - np.sin(x) * np.sin(y_samples) + 0.5 <= 0)
        y_samples = y_samples[y_samples_index]
        true_max_val[i] = np.max(np.cos(2 * x) * np.cos(y_samples) + np.sin(x))
        # use scipy to compute the ground truth
        # import pdb
        # pdb.set_trace()
    # compute the fitting error
    L2_error = np.mean(np.power(true_max_val - pred_max, 2))
    print(L2_error)
    plt.plot(ctx_array, pred_max, label='bayesian')
    plt.plot(ctx_array, true_max_val, label='true')
    plt.legend()
    plt.xlabel('x', fontsize=16)
    plt.ylabel('$\\max_a f(x, a)$', fontsize=16, rotation=0)
    plt.savefig('build/cbo_2.png')
    plt.show()