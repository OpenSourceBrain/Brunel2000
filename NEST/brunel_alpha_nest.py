#
# This file is taken from the NEST standard examples
#
# Copyright (C) 2004 The NEST Initiative
#

import nest

import numpy as np
import scipy.special as sp

import time


def LambertWm1(x):
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)

    # time of maximum
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude
    return (np.exp(1.0) / (tauSyn * CMem * b) *
            ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b -
             t_max * np.exp(-t_max / tauSyn)))



def runBrunelNetwork(g=5., eta=2., dt = 0.1, simtime = 1000.0, delay = 1.5, epsilon = 0.1, order = 2500, N_rec = 50, save=False, simulator_name='nest',jnml_simulator=None):

    nest.ResetKernel()
    startbuild = time.time()

    dt = dt    # the resolution in ms
    simtime = simtime  # Simulation time in ms
    delay = delay    # synaptic delay in ms

    g = g  # ratio inhibitory weight/excitatory weight
    eta = eta  # external rate relative to threshold rate
    epsilon = epsilon  # connection probability


    order = order
    NE = 4 * order  # number of excitatory neurons
    NI = 1 * order  # number of inhibitory neurons
    N_neurons = NE + NI   # number of neurons in total
    N_rec = N_rec      # record from 50 neurons


    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI + CE)      # total number of synapses per neuron


    tauSyn = 0.5  # synaptic time constant in ms
    tauMem = 20.0  # time constant of membrane potential in ms
    CMem = 250.0  # capacitance of membrane in in pF
    theta = 20.0  # membrane threshold potential in mV
    neuron_params = {"C_m":        CMem,
                     "tau_m":      tauMem,
                     "tau_syn_ex": tauSyn,
                     "tau_syn_in": tauSyn,
                     "t_ref":      2.0,
                     "E_L":        0.0,
                     "V_reset":    0.0,
                     "V_m":        0.0,
                     "V_th":       theta}
    J = 0.1        # postsynaptic amplitude in mV
    J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
    J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
    J_in = -g * J_ex    # amplitude of inhibitory postsynaptic current


    nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE


    nest.resolution = dt
    nest.print_time = True
    nest.overwrite_files = True

    print("Building network")

    nest.SetDefaults("iaf_psc_alpha", neuron_params)
    nest.SetDefaults("poisson_generator", {"rate": p_rate})

    nodes_ex = nest.Create("iaf_psc_alpha", NE)
    nodes_in = nest.Create("iaf_psc_alpha", NI)
    nodes_all = nodes_ex+nodes_in
    noise = nest.Create("poisson_generator")
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    all_spikes  = nest.Create("spike_recorder")


    espikes.set(label="brunel-py-ex", record_to="ascii")
    ispikes.set(label="brunel-py-in", record_to="ascii")


    print("Connecting devices")

    nest.CopyModel("static_synapse", "excitatory", {
                   "weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory", {
                   "weight": J_in, "delay": delay})

    nest.Connect(noise, nodes_ex, syn_spec="excitatory")
    nest.Connect(noise, nodes_in, syn_spec="excitatory")

    nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
    nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

    nest.Connect(nodes_all, all_spikes, syn_spec="excitatory")

    print("Connecting network")

    np.random.seed(1234)

    print("Excitatory connections")

    ###############################################################################
    # Connecting the excitatory population to all neurons using the pre-defined
    # excitatory synapse. Beforehand, the connection parameter are defined in a
    # dictionary. Here we use the connection rule ``fixed_indegree``,
    # which requires the definition of the indegree. Since the synapse
    # specification is reduced to assigning the pre-defined excitatory synapse it
    # suffices to insert a string.

    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")

    print("Inhibitory connections")

    ###############################################################################
    # Connecting the inhibitory population to all neurons using the pre-defined
    # inhibitory synapse. The connection parameter as well as the synapse
    # parameter are defined analogously to the connection from the excitatory
    # population defined above.

    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")


    endbuild = time.time()


    print("Simulating")

    nest.Simulate(simtime)
    endsimulate = time.time()

    events_ex = nest.GetStatus(espikes, "n_events")[0]
    events_in = nest.GetStatus(ispikes, "n_events")[0]

    rate_ex = events_ex / simtime * 1000.0 / N_rec
    rate_in = events_in / simtime * 1000.0 / N_rec

    num_synapses = nest.GetDefaults("excitatory")["num_connections"] +\
        nest.GetDefaults("inhibitory")["num_connections"]

    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild

    print("Brunel network simulation (Python)")
    print("Number of neurons : {0}".format(N_neurons))
    print("Number of synapses: {0}".format(num_synapses))
    print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
    print("       Inhibitory : {0}".format(int(CI * N_neurons)))
    print("Excitatory rate   : %.2f Hz" % rate_ex)
    print("Inhibitory rate   : %.2f Hz" % rate_in)
    print("Building time     : %.2f s" % build_time)
    print("Simulation time   : %.2f s" % sim_time)


    #import nest.raster_plot
    #nest.raster_plot.from_device(espikes, hist=True)
    
    return all_spikes

if __name__ == '__main__':

    simtime = 1000.0
    order = 2500

    eta         = 2.0     # rel rate of external input
    g           = 5.0

    runBrunelNetwork(g=g, eta=eta, simtime = simtime, order = order, save=True)