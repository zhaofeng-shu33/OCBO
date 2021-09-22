import numpy as np
import matplotlib.pyplot as plt

from OCBO.cstrats.profile_cts import ContinuousMultiTaskTS
from OCBO.cstrats import copts
from dragonfly.utils.option_handler import load_options

from OCBO.util.misc_util import uniform_draw

def black_box_function_1(vec):
    x, y = vec
    func_val = np.cos(2 * x) * np.cos(y) + np.sin(x)
    # constraint_val = np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) + 0.5
    # return (-func_val, constraint_val)
    return -func_val

function = black_box_function_1
domain = [[0, 6], [0, 6]]
ctx_dim = 1
max_capital = 100
init_capital = 10

options = load_options(copts)
model = ContinuousMultiTaskTS(function, domain, ctx_dim, options, eval_set=True, is_synthetic=False)
init_pts = list(uniform_draw(domain, init_capital))
# switch off the hyper-parameter tuning of GP
histories = model.optimize(max_capital, init_pts=init_pts, pre_tune=True)
# print(histories)
# plotting the result

ctx_array = np.linspace(0, 6)
n = ctx_array.shape[0]
action = np.zeros(n)
for i in range(n):
    ctx_plus_action = model._get_ctx_improvement([ctx_array[i]], predict=True)
    # extract the action
    action[i] = ctx_plus_action[-1]
    # import pdb
    # pdb.set_trace()
plt.plot(ctx_array, action)
plt.show()