import nest
from scipy.special import lambertw

import numpy
from numpy import exp

import time


def computePSPnorm(tauMem, CMem, tauSyn):
    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)
    t_max = 1.0 / b * (-lambertw(-exp(-1.0 / a) / a, k=-1).real - 1.0 / a)
    return exp(1.0) / (tauSyn * CMem * b) * ((exp(-t_max / tauMem) - exp(-t_max / tauSyn)) / b - t_max * exp(-t_max / tauSyn))



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
    J_unit = computePSPnorm(tauMem, CMem, tauSyn)
    J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
    J_in = -g * J_ex    # amplitude of inhibitory postsynaptic current


    nu_th = (theta * CMem) / (J_ex * CE * numpy.exp(1) * tauMem * tauSyn)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE


    nest.SetKernelStatus(
        {"resolution": dt, "print_time": True, "overwrite_files": True})

    print("Building network")

    nest.SetDefaults("iaf_psc_alpha", neuron_params)
    nest.SetDefaults("poisson_generator", {"rate": p_rate})

    nodes_ex = nest.Create("iaf_psc_alpha", NE)
    nodes_in = nest.Create("iaf_psc_alpha", NI)
    nodes_all = nodes_ex+nodes_in
    noise = nest.Create("poisson_generator")
    espikes = nest.Create("spike_detector")
    ispikes = nest.Create("spike_detector")
    all_spikes  = nest.Create("spike_detector")

    nest.SetStatus(espikes, [{"label": "brunel-py-ex",
                              "withtime": True,
                              "withgid": True,
                              "to_file": save}])

    nest.SetStatus(ispikes, [{"label": "brunel-py-in",
                              "withtime": True,
                              "withgid": True,
                              "to_file": save}])

    nest.SetStatus(all_spikes,[{"label": "brunel-py-all",
                             "withtime": True,
                             "withgid": True,
                             "to_file": False}])

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

    numpy.random.seed(1234)

    sources_ex = numpy.random.random_integers(1, NE, (N_neurons, CE))
    sources_in = numpy.random.random_integers(NE + 1, N_neurons, (N_neurons, CI))

    for n in range(N_neurons):
        nest.Connect(list(sources_ex[n]), [n + 1], syn_spec="excitatory")

    for n in range(N_neurons):
        nest.Connect(list(sources_in[n]), [n + 1], syn_spec="inhibitory")


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