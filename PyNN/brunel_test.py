"""
A scaled down version of the Brunel model useful for testing (see OMV files: .test.*)

"""

from brunel08 import runBrunelNetwork

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]
simtime = 1000
order = 100

eta         = 2.0     # rel rate of external input
g           = 5.0

runBrunelNetwork(g=g, eta=eta, simtime = simtime, order = order, save=True, simulator_name=simulator_name,N_rec=500)
