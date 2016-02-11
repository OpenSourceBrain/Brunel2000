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

'''
Assigning the simulation parameters to variables.
'''

dt      = 0.1    # the resolution in ms
simtime = 1000.0 # Simulation time in ms
delay   = 1.5    # synaptic delay in ms

'''
Definition of the parameters crucial for asynchronous irregular firing
of the neurons.
'''

#g       = 5.0  # ratio inhibitory weight/excitatory weight
#eta     = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

'''
Definition of the number of neurons in the network and the number of
neuron recorded from
'''

order     = 2500
NE        = 4*order # number of excitatory neurons
NI        = 1*order # number of inhibitory neurons
N_neurons = NE+NI   # number of neurons in total
N_rec     = 50     # record from 50 neurons

def _spikingNetwork_(g=5., eta=2.):
    nest.ResetKernel()

    '''
    Assigning the current time to a variable in order to determine the
    build time of the network.
    '''

    startbuild = time.time()


    '''
    Definition of connectivity parameter
    '''

    CE    = int(epsilon*NE) # number of excitatory synapses per neuron
    CI    = int(epsilon*NI) # number of inhibitory synapses per neuron
    C_tot = int(CI+CE)      # total number of synapses per neuron

    '''
    Initialization of the parameters of the integrate and fire neuron and
    the synapses. The parameter of the neuron are stored in a dictionary.
    '''

    tauMem = 20.0   # time constant of membrane potential in ms
    theta  = 20.0   # membrane threshold potential in mV
    neuron_params= {"C_m":        1.0,
                    "tau_m":      tauMem,
                    "t_ref":      2.0,
                    "E_L":        0.0,
                    "V_reset":    0.0,
                    "V_m":        0.0,
                    "V_th":       theta}
    J     = 0.1     # postsynaptic amplitude in mV
    J_ex  = J       # amplitude of excitatory postsynaptic potential
    J_in  = -g*J_ex # amplitude of inhibitory postsynaptic potential


    '''
    Definition of threshold rate, which is the external rate needed to fix
    the membrane potential around its threshold, the external firing rate
    and the rate of the poisson generator which is multiplied by the
    in-degree CE and converted to Hz by multiplication by 1000.
    '''

    nu_th  = theta/(J*CE*tauMem)
    nu_ex  = eta*nu_th
    p_rate = 1000.0*nu_ex*CE

    '''
    Configuration of the simulation kernel by the previously defined time
    resolution used in the simulation. Setting "print_time" to True prints
    the already processed simulation time as well as its percentage of the
    total simulation time.
    '''

    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True, 'local_num_threads': 8})

    print("Building network")

    '''
    Configuration of the model `iaf_psc_delta` and `poisson_generator`
    using SetDefaults(). This function expects the model to be the
    inserted as a string and the parameter to be specified in a
    dictionary. All instances of theses models created after this point
    will have the properties specified in the dictionary by default.
    '''

    nest.SetDefaults("iaf_psc_delta", neuron_params)
    nest.SetDefaults("poisson_generator",{"rate": p_rate})

    '''
    Creation of the nodes using `Create`. We store the returned handles in
    variables for later reference. Here the excitatory and inhibitory, as
    well as the poisson generator and two spike detectors. The spike
    detectors will later be used to record excitatory and inhibitory
    spikes.
    '''

    nodes_ex = nest.Create("iaf_psc_delta",NE)
    nodes_in = nest.Create("iaf_psc_delta",NI)
    nodes_all = nodes_ex+nodes_in
    noise    = nest.Create("poisson_generator")
    espikes  = nest.Create("spike_detector")
    ispikes  = nest.Create("spike_detector")
    all_spikes  = nest.Create("spike_detector")

    '''
    Configuration of the spike detectors recording excitatory and
    inhibitory spikes using `SetStatus`, which expects a list of node
    handles and a list of parameter dictionaries. Setting the variable
    "to_file" to True ensures that the spikes will be recorded in a .gdf
    file starting with the string assigned to label. Setting "withtime"
    and "withgid" to True ensures that each spike is saved to file by
    stating the gid of the spiking neuron and the spike time in one line.
    '''

    nest.SetStatus(espikes,[{"label": "brunel-py-ex",
                             "withtime": True,
                             "withgid": True,
                             "to_file": False}])

    nest.SetStatus(ispikes,[{"label": "brunel-py-in",
                             "withtime": True,
                             "withgid": True,
                             "to_file": False}])

    nest.SetStatus(all_spikes,[{"label": "brunel-py-all",
                             "withtime": True,
                             "withgid": True,
                             "to_file": False}])

    print("Connecting devices")

    '''
    Definition of a synapse using `CopyModel`, which expects the model
    name of a pre-defined synapse, the name of the customary synapse and
    an optional parameter dictionary. The parameters defined in the
    dictionary will be the default parameter for the customary
    synapse. Here we define one synapse for the excitatory and one for the
    inhibitory connections giving the previously defined weights and equal
    delays.
    '''

    nest.CopyModel("static_synapse","excitatory",{"weight":J_ex, "delay":delay})
    nest.CopyModel("static_synapse","inhibitory",{"weight":J_in, "delay":delay})

    '''
    Connecting the previously defined poisson generator to the excitatory
    and inhibitory neurons using the excitatory synapse. Since the poisson
    generator is connected to all neurons in the population the default
    rule ('all_to_all') of Connect() is used. The synaptic properties are
    inserted via syn_spec which expects a dictionary when defining
    multiple variables or a string when simply using a pre-defined
    synapse.
    '''

    nest.Connect(noise,nodes_ex, syn_spec="excitatory")
    nest.Connect(noise,nodes_in, syn_spec="excitatory")

    '''
    Connecting the first N_rec nodes of the excitatory and inhibitory
    population to the associated spike detectors using excitatory
    synapses. Here the same shortcut for the specification of the synapse
    as defined above is used.
    '''

    nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
    nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

    nest.Connect(nodes_all, all_spikes, syn_spec="excitatory")

    print("Connecting network")

    print("Excitatory connections")

    '''
    Connecting the excitatory population to all neurons using the
    pre-defined excitatory synapse. Beforehand, the connection parameter
    are defined in a dictionary. Here we use the connection rule
    'fixed_indegree', which requires the definition of the indegree. Since
    the synapse specification is reduced to assigning the pre-defined
    excitatory synapse it suffices to insert a string.
    '''

    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex+nodes_in, conn_params_ex, "excitatory")

    print("Inhibitory connections")

    '''
    Connecting the inhibitory population to all neurons using the
    pre-defined inhibitory synapse. The connection parameter as well as
    the synapse paramtere are defined analogously to the connection from
    the excitatory population defined above.
    '''

    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex+nodes_in, conn_params_in, "inhibitory")

    '''
    Storage of the time point after the buildup of the network in a
    variable.
    '''

    endbuild=time.time()

    '''
    Simulation of the network.
    '''

    print("Simulating")

    nest.Simulate(simtime)

    '''
    Storage of the time point after the simulation of the network in a
    variable.
    '''

    endsimulate= time.time()

    '''
    Reading out the total number of spikes received from the spike
    detector connected to the excitatory population and the inhibitory
    population.
    '''

    events_ex = nest.GetStatus(espikes,"n_events")[0]
    events_in = nest.GetStatus(ispikes,"n_events")[0]

    '''
    Calculation of the average firing rate of the excitatory and the
    inhibitory neurons by dividing the total number of recorded spikes by
    the number of neurons recorded from and the simulation time. The
    multiplication by 1000.0 converts the unit 1/ms to 1/s=Hz.
    '''

    rate_ex   = events_ex/simtime*1000.0/N_rec
    rate_in   = events_in/simtime*1000.0/N_rec

    '''
    Reading out the number of connections established using the excitatory
    and inhibitory synapse model. The numbers are summed up resulting in
    the total number of synapses.
    '''

    num_synapses = nest.GetDefaults("excitatory")["num_connections"]+\
    nest.GetDefaults("inhibitory")["num_connections"]

    '''
    Establishing the time it took to build and simulate the network by
    taking the difference of the pre-defined time variables.
    '''

    build_time = endbuild-startbuild
    sim_time   = endsimulate-endbuild

    '''
    Printing the network properties, firing rates and building times.
    '''

    print("Brunel network simulation (Python)")
    print("Number of neurons : {0}".format(N_neurons))
    print("Number of synapses: {0}".format(num_synapses))
    print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
    print("       Inhibitory : {0}".format(int(CI * N_neurons)))
    print("Excitatory rate   : %.2f Hz" % rate_ex)
    print("Inhibitory rate   : %.2f Hz" % rate_in)
    print("Building time     : %.2f s" % build_time)
    print("Simulation time   : %.2f s" % sim_time)

    '''
    Plot a raster of the excitatory neurons and a histogram.
    '''

    #nest.raster_plot.from_device(espikes, hist=True)

    return all_spikes

###

N = NE+NI

## ratio inhibitory weight/excitatory weight
g_rng = np.arange(3, 9, .5)
## external rate relative to threshold rate
eta_rng = np.arange(.5, 4., .5)

sim_run = 1

if sim_run:
    results = {}
    ISIcv = {}
    FF = {}
    all_rates = {}
    for i1, g in enumerate(g_rng):
        for i2, eta in enumerate(eta_rng):
            print('')
            print('########## (g, eta): ', g, ' ,', eta)

            all_spikes = _spikingNetwork_(g=g, eta=eta)

            ta0 = time.time()

            spd_all = nest.GetStatus(all_spikes)[0]['events']

            all_rates[g, eta] = np.histogram(spd_all['senders'], range=(1,N), bins=N)[0]

            binw=1. #ms
            pop_rate = np.histogram(spd_all['times'], range=(0,simtime), bins=simtime/binw)[0] / (binw/1000.)/ N

            FF[g, eta] = np.var(pop_rate) / np.mean(pop_rate)

            isi_cv = np.zeros(N)
            for ii in range(N):
                nid = ii+1
                spids = np.where(spd_all['senders'] == nid)
                isi = np.diff(spd_all['times'][spids])
                isi_cv[ii] = np.std(isi)/ np.mean(isi)

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
            I[i1,i2] = np.mean(ISIcv[g,eta])
            S[i1,i2] = FF[g,eta]

            Rexc[i1,i2] = np.mean(all_rates[g,eta][0:NE])
            Rinh[i1,i2] = np.mean(all_rates[g,eta][NE:])

def _plot_(X, sbplt=111, ttl=[]):
    ax = pl.subplot(sbplt)
    pl.title(ttl)
    pl.imshow(X, origin='lower', interpolation='none')
    pl.xlabel('g')
    pl.ylabel(r'$\nu_{ext} / \nu_{thr}$')
    ax.set_xticks(range(0,len(g_rng))); ax.set_xticklabels(g_rng)
    ax.set_yticks(range(0,len(eta_rng))); ax.set_yticklabels(eta_rng)
    pl.colorbar()

pl.figure(figsize=(16,8))

_plot_(Rexc.T, 221, 'Rates Exc')
_plot_(Rinh.T, 222, 'Rates Inh')

_plot_(S.T, 223, 'Synchrony (FF)')
_plot_(I.T, 224, 'Irregularity (ISI CV)')

pl.subplots_adjust(wspace=.3, hspace=.3)

pl.show()
