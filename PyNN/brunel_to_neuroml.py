"""
A scaled down version of the Brunel model converted to NeuroML 2

"""

from brunel08 import runBrunelNetwork
import shutil

simulator_name = 'neuroml'
simtime = 500
order = 10

eta         = 2.0     # rel rate of external input
g           = 5.0

runBrunelNetwork(g=g, eta=eta, simtime = simtime, order = order, save=True, simulator_name=simulator_name)

shutil.copy('PyNN_NeuroML2_Export.nml','../NeuroML2/BrunelFromPyNN.net.nml')
shutil.copy('LEMS_Sim_PyNN_NeuroML2_Export.xml','../NeuroML2/LEMS_BrunelFromPyNN.xml')




