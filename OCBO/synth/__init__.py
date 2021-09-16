from OCBO.synth.function_generator import add_funcs
from OCBO.synth.oned import oneD_functions
from OCBO.synth.twod import twod_functions
from OCBO.synth.sixd import sixd_functions

synth_functions = oneD_functions + twod_functions + sixd_functions + add_funcs
