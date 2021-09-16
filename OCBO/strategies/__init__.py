from argparse import Namespace

from OCBO.strategies.agnostic_opt import agn_strats
from OCBO.strategies.corr_opt import corr_strats, corr_args
from OCBO.strategies.multi_opt import multi_opt_args
from OCBO.strategies.random_strat import RandomOpt, JointRandom
from OCBO.strategies.mei import MEI
from OCBO.strategies.mts import MTS
from OCBO.strategies.joint_agnostic_opt import ja_strats
from OCBO.strategies.joint_mei import JointMEI
from OCBO.strategies.joint_mts import JointMTS

discrete_strategies = [Namespace(name='Random', impl=RandomOpt),
                       Namespace(impl=MTS, name=MTS.get_opt_method_name()),
                       Namespace(impl=MEI, name=MEI.get_opt_method_name())] \
                      + agn_strats \
                      + corr_strats

joint_strategies = [Namespace(impl=JointRandom,
                              name=JointRandom.get_opt_method_name()),
                    Namespace(impl=JointMTS,
                              name=JointMTS.get_opt_method_name()),
                    Namespace(impl=JointMEI,
                              name=JointMEI.get_opt_method_name())] \
                   + ja_strats

strategies = discrete_strategies + joint_strategies

strat_args =  corr_args + multi_opt_args
