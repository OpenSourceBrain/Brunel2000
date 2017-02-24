"""
A scaled down version of the Brunel model converted to NeuroML 2

"""

from brunel08 import runBrunelNetwork
import shutil

simulator_name = 'neuroml'
simtime = 1000
order = 5

eta         = 1     # rel rate of external input
g           = 4

import logging
logging.basicConfig(level=logging.DEBUG, format="%(name)-19s %(levelname)-5s - %(message)s")

runBrunelNetwork(g=g, 
                 eta=eta, 
                 simtime = simtime, 
                 dt = 0.025, # for jLEMS
                 order = order, 
                 save=True, 
                 simulator_name=simulator_name,
                 extra={'reference':'BrunelFromPyNN'})

shutil.copy('BrunelFromPyNN.net.nml','../NeuroML2')
#shutil.copy('BrunelFromPyNN.net.nml','../NeuroML2')
shutil.copy('LEMS_Sim_BrunelFromPyNN.xml','../NeuroML2/LEMS_Sim_BrunelFromPyNN.xml')




