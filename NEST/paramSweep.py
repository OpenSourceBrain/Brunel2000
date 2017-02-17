# -*- coding: utf-8 -*-
#
# brunel_delta_nest.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

'''
Random balanced network (delta synapses)
----------------------------------------

This script simulates an excitatory and an inhibitory population on
the basis of the network used in

Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
183–208 (2000).

When connecting the network customary synapse models are used, which
allow for querying the number of created synapses. Using spike
detectors the average firing rates of the neurons in the populations
are established. The building as well as the simulation time of the
network are recorded.
'''

'''
Importing all necessary modules for simulation, analysis and plotting.
'''
import pickle
import sys

import nest
import nest.raster_plot

import time
import numpy as np
import pylab as pl


def runParameterSweep(runBrunelNetwork, label, simtime = 1000.0, order = 2500, simulator_name=None, jnml_simulator=None, quick=False):
    ## ratio inhibitory weight/excitatory weight
    g_rng = np.arange(3, 9, .5)
    ## external rate relative to threshold rate
    eta_rng = np.arange(.5, 4., .5)
    
    if quick:
        g_rng = np.arange(3, 9, 1)
        eta_rng = np.arange(.5, 4., 1)
        #g_rng = np.arange(5, 6, 1)
        #eta_rng = np.arange(2, 3, 1)
        
    if not simulator_name:
        simulator_name = 'NEST'

    sim_run = 1

    NE = order*4
    N = order*5

    if sim_run:
        results = {}
        ISIcv = {}
        FF = {}
        all_rates = {}
        count = 1
        for i1, g in enumerate(g_rng):
            for i2, eta in enumerate(eta_rng):
                print('\n\n###########################################################################################')
                print('############# (Running with params: g=%s, eta=%s; %s/%s): '% (g, eta, count, len(g_rng)*len(eta_rng)))
                count+=1
                dt = 0.1
                if jnml_simulator:
                    dt=0.025
                all_spikes = runBrunelNetwork(g=g, 
                                              eta=eta, 
                                              simtime=simtime, 
                                              dt = dt,
                                              order=order, 
                                              N_rec=NE, 
                                              simulator_name=simulator_name,
                                              jnml_simulator=jnml_simulator)

                ta0 = time.time()
                if 'pynn' in label:
                    spd_all = {}
                    spd_all['senders'] = np.array(all_spikes['senders'])
                    spd_all['times'] = np.array(all_spikes['times'])
                    
                else:
                    spd_all = nest.GetStatus(all_spikes)[0]['events']
                    
                
                print("All spike data: %s"%spd_all)

                all_rates[g, eta] = np.histogram(spd_all['senders'], range=(1,N), bins=N)[0] / (simtime/1e3)
                print("All rates: %s"%all_rates[g, eta])

                binw=1. #ms
                pop_rate = np.histogram(spd_all['times'], range=(0,simtime), bins=simtime/binw)[0] / (binw/1000.)/ N
                #print("pop_rate: %s"%pop_rate)

                FF[g, eta] = np.var(pop_rate) / np.mean(pop_rate)

                isi_cv = np.zeros(N)
                for ii in range(N):
                    nid = ii+1
                    spids = np.where(spd_all['senders'] == nid)
                    isi = np.diff(spd_all['times'][spids])
                    isi_cv[ii] = np.std(isi)/ np.mean(isi)
                    #print("Cell %i has isi: %s; cv: %s"%(nid,isi,isi_cv[ii]))

                ISIcv[g, eta] = isi_cv

                taf = time.time()
                
                print("Analysis time   : %.2f s" % (taf-ta0))


        results['FF'] = FF
        results['ISIcv'] = ISIcv
        results['all_rates'] = all_rates
        
        fl = open('results', 'wb')
        pickle.dump(results, fl)
        fl.close()
    else:
        fl = open('results', 'rb')
        results = pickle.load(fl)
        fl.close()

        FF = results['powspect']
        ISIcv = results['powmean']
        all_rates = results['all_rates']

    ###

    # synchrony
    S = np.zeros((len(g_rng), len(eta_rng)))
    # irregularity
    I = np.zeros((len(g_rng), len(eta_rng)))

    Rexc = np.zeros((len(g_rng), len(eta_rng)))
    Rinh = np.zeros((len(g_rng), len(eta_rng)))

    for i1, g in enumerate(g_rng):
            for i2, eta in enumerate(eta_rng):
                #print("- %s"%(ISIcv[g,eta]))
                I[i1,i2] = np.mean(ISIcv[g,eta])
                S[i1,i2] = FF[g,eta]

                Rexc[i1,i2] = np.mean(all_rates[g,eta][0:NE])
                Rinh[i1,i2] = np.mean(all_rates[g,eta][NE:])
                print("g=%s, eta=%s: Rexc=%s, Rinh=%s, S=%s, I=%s"%(g,eta,Rexc[i1,i2],Rinh[i1,i2],S[i1,i2],I[i1,i2]))

    def _plot_(X, sbplt=111, ttl=[]):
        ax = pl.subplot(sbplt)
        pl.title(ttl)
        pl.imshow(X, origin='lower', interpolation='none')
        pl.xlabel('g')
        pl.ylabel(r'$\nu_{ext} / \nu_{thr}$')
        ax.set_xticks(range(0,len(g_rng))); ax.set_xticklabels(g_rng)
        ax.set_yticks(range(0,len(eta_rng))); ax.set_yticklabels(eta_rng)
        pl.colorbar()

    fig = pl.figure(figsize=(16,8))
    info = "%s (%s) %i exc, %i inh cells, %s ms"%(simulator_name.upper(),label,NE, N-NE, simtime)
    
    fig.canvas.set_window_title(info)
    pl.suptitle(info)

    _plot_(Rexc.T, 221, 'Rates Exc (Hz)')
    _plot_(Rinh.T, 222, 'Rates Inh (Hz)')

    _plot_(S.T, 223, 'Synchrony (FF)')
    _plot_(I.T, 224, 'Irregularity (ISI CV)')

    pl.subplots_adjust(wspace=.3, hspace=.3)


    pl.savefig('%s_%s_N%s_%sms.png'%(simulator_name.upper(), label,N,simtime), bbox_inches='tight')
    print("Finished: "+info)
    pl.show()
    
    
    
if __name__ == '__main__':
    
    simtime = 1000.0
    order = 500
    

    from brunel_delta_nest import runBrunelNetwork as runBrunelNetworkDelta
    from brunel_alpha_nest import runBrunelNetwork as runBrunelNetworkAlpha
    
    sys.path.append("../PyNN")
    from brunel08 import runBrunelNetwork as runBrunelNetworkPyNN
    
    #runParameterSweep(runBrunelNetworkDelta, "delta", simtime=1000, order=100, quick=True)
    #runParameterSweep(runBrunelNetworkDelta, "delta", simtime=simtime, order=order)
    #runParameterSweep(runBrunelNetworkAlpha, "alpha", simtime=1000, order=100, quick=True)
    #runParameterSweep(runBrunelNetworkPyNN, "pynn_nest", simtime=100, order=10, simulator_name='nest', quick=True)
    #runParameterSweep(runBrunelNetworkPyNN, "pynn_brian", simtime=1000, order=50, simulator_name='brian', quick=True)
    #runParameterSweep(runBrunelNetworkPyNN, "pynn_neuron", simtime=1000, order=100, simulator_name='neuron', quick=True)
    #runParameterSweep(runBrunelNetworkPyNN, "pynn_nest", simtime=1000, order=100, simulator_name='nest', quick=True)
    #runParameterSweep(runBrunelNetworkPyNN, "pynn_neuroml", simtime=1000, order=10, simulator_name='neuroml', jnml_simulator='jNeuroML', quick=True)
    runParameterSweep(runBrunelNetworkPyNN, "pynn_neuroml", simtime=1000, order=20, simulator_name='neuroml', jnml_simulator='jNeuroML_NEURON', quick=True)
