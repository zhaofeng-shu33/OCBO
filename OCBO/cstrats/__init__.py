"""
Continuous strategies.
"""
from argparse import Namespace

from OCBO.cstrats.agn_cts import agn_strats, agn_args
from OCBO.cstrats.cts_opt import cts_opt_args
from OCBO.cstrats.postmax_cts import pm_strats, pm_args
from OCBO.cstrats.profile_cts import prof_strats, prof_args
from OCBO.cstrats.rand_cts import RandOpt

cstrats = [Namespace(impl=RandOpt, name=RandOpt.get_strat_name())] \
        + pm_strats \
        + prof_strats \
        + agn_strats

copts = cts_opt_args + pm_args + prof_args + agn_args
