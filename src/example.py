from cstrats.profile_cts import ContinuousMultiTaskTS
from cstrats import copts
from dragonfly.utils.option_handler import load_options
from synth.twod import branin
from util.misc_util import uniform_draw

from argparse import Namespace
function = branin
domain = [[0, 1], [0, 1]]
ctx_dim = 1
max_capital = 20
init_capital = 5
init_pts = 1
options = load_options(copts)
method = ContinuousMultiTaskTS(function, domain, ctx_dim, options)
init_pts = list(uniform_draw(domain, init_capital))
histories = method.optimize(max_capital, init_pts=init_pts)
print(histories)
# extract sampling points from history and save it in mat format


