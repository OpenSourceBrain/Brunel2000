"""
Conversion of the Brunel network implemented in nest-1.0.13/examples/brunel.sli
to use PyNN.

Brunel N (2000) Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. J Comput Neurosci 8:183-208

Andrew Davison, UNIC, CNRS
May 2006

"""

from pyNN.utility import get_script_args, Timer, ProgressBar
from pyNN.random import NumpyRNG, RandomDistribution
from neo.io import PyNNTextIO


def runBrunelNetwork(g=5., 
                     eta=2., 
                     dt = 0.1, 
                     simtime = 1000.0, 
                     delay = 1.5, 
                     epsilon = 0.1, 
                     order = 2500, 
                     N_rec = 50,
                     N_rec_v = 2, 
                     save=False, 
                     simulator_name='nest',
                     jnml_simulator=None,
                     extra = {}):

    exec("from pyNN.%s import *" % simulator_name) in globals()
    
    timer = Timer()

    # === Define parameters ========================================================

    downscale   = 1       # scale number of neurons down by this factor
                          # scale synaptic weights up by this factor to
                          # obtain similar dynamics independent of size
    order       = order   # determines size of network:
                          # 4*order excitatory neurons
                          # 1*order inhibitory neurons
    Nrec        = N_rec   # number of neurons to record from, per population
    epsilon     = epsilon # connectivity: proportion of neurons each neuron projects to

    # Parameters determining model dynamics, cf Brunel (2000), Figs 7, 8 and Table 1
    # here: Case C, asynchronous irregular firing, ~35 Hz
    eta         = eta     # rel rate of external input
    g           = g       # rel strength of inhibitory synapses
    J           = 0.1     # synaptic weight [mV]
    delay       = delay   # synaptic delay, all connections [ms]

    # single neuron parameters
    tauMem      = 20.0    # neuron membrane time constant [ms]
    tauSyn      = 0.1     # synaptic time constant [ms]
    tauRef      = 2.0     # refractory time [ms]
    U0          = 0.0     # resting potential [mV]
    theta       = 20.0    # threshold

    # simulation-related parameters
    simtime     = simtime   # simulation time [ms]
    dt          = dt     # simulation step length [ms]

    # seed for random generator used when building connections
    connectseed = 12345789
    use_RandomArray = True  # use Python rng rather than NEST rng

    # seed for random generator(s) used during simulation
    kernelseed  = 43210987

    # === Calculate derived parameters =============================================

    # scaling: compute effective order and synaptic strength
    order_eff = int(float(order)/downscale)
    J_eff     = J*downscale

    # compute neuron numbers
    NE = int(4*order_eff)  # number of excitatory neurons
    NI = int(1*order_eff)  # number of inhibitory neurons
    N  = NI + NE           # total number of neurons

    # compute synapse numbers
    CE   = int(epsilon*NE)  # number of excitatory synapses on neuron
    CI   = int(epsilon*NI)  # number of inhibitory synapses on neuron
    C    = CE + CI          # total number of internal synapses per n.
    Cext = CE               # number of external synapses on neuron

    # synaptic weights, scaled for alpha functions, such that
    # for constant membrane potential, charge J would be deposited
    fudge = 0.00041363506632638  # ensures dV = J at V=0

    # excitatory weight: JE = J_eff / tauSyn * fudge
    JE = (J_eff/tauSyn)*fudge

    # inhibitory weight: JI = - g * JE
    JI = -g*JE

    # threshold, external, and Poisson generator rates:
    nu_thresh = theta/(J_eff*CE*tauMem)
    nu_ext    = eta*nu_thresh     # external rate per synapse
    p_rate    = 1000*nu_ext*Cext  # external input rate per neuron (Hz)

    # number of synapses---just so we know
    Nsyn = (C+1)*N + 2*Nrec  # number of neurons * (internal synapses + 1 synapse from PoissonGenerator) + 2synapses" to spike detectors

    # put cell parameters into a dict
    cell_params = {'tau_m'      : tauMem,
                   'tau_syn_E'  : tauSyn,
                   'tau_syn_I'  : tauSyn,
                   'tau_refrac' : tauRef,
                   'v_rest'     : U0,
                   'v_reset'    : U0,
                   'v_thresh'   : theta,
                   'cm'         : 0.001}     # (nF)

    # === Build the network ========================================================

    # clear all existing network elements and set resolution and limits on delays.
    # For NEST, limits must be set BEFORE connecting any elements

    #extra = {'threads' : 2}

    rank = setup(timestep=dt, max_delay=delay, **extra)
    print("rank =", rank)
    np = num_processes()
    print("np =", np)
    import socket
    host_name = socket.gethostname()
    print("Host #%d is on %s" % (rank+1, host_name))

    if 'threads' in extra:
        print("%d Initialising the simulator with %d threads..." % (rank, extra['threads']))
    else:
        print("%d Initialising the simulator with single thread..." % rank)

    # Small function to display information only on node 1
    def nprint(s):
        if rank == 0:
            print(s)

    timer.start()  # start timer on construction

    print("%d Setting up random number generator" % rank)
    rng = NumpyRNG(kernelseed, parallel_safe=True)

    print("%d Creating excitatory population with %d neurons." % (rank, NE))
    celltype = IF_curr_alpha(**cell_params)
    celltype.default_initial_values['v'] = U0 # Setting default init v, useful for NML2 export
    E_net = Population(NE, celltype, label="E_net")

    print("%d Creating inhibitory population with %d neurons." % (rank, NI))
    I_net = Population(NI, celltype, label="I_net")

    print("%d Initialising membrane potential to random values between %g mV and %g mV." % (rank, U0, theta))
    uniformDistr = RandomDistribution('uniform', low=U0, high=theta, rng=rng)
    E_net.initialize(v=uniformDistr)
    I_net.initialize(v=uniformDistr)

    print("%d Creating excitatory Poisson generator with rate %g spikes/s." % (rank, p_rate))
    source_type = SpikeSourcePoisson(rate=p_rate)
    expoisson = Population(NE, source_type, label="expoisson")

    print("%d Creating inhibitory Poisson generator with the same rate." % rank)
    inpoisson = Population(NI, source_type, label="inpoisson")

    # Record spikes
    print("%d Setting up recording in excitatory population." % rank)
    E_net.record('spikes')
    if N_rec_v>0:
        E_net[0:min(NE,N_rec_v)].record('v')

    print("%d Setting up recording in inhibitory population." % rank)
    I_net.record('spikes')
    if N_rec_v>0:
        I_net[0:min(NI,N_rec_v)].record('v')

    progress_bar = ProgressBar(width=20)
    connector = FixedProbabilityConnector(epsilon, rng=rng, callback=progress_bar)
    E_syn = StaticSynapse(weight=JE, delay=delay)
    I_syn = StaticSynapse(weight=JI, delay=delay)
    ext_Connector = OneToOneConnector(callback=progress_bar)
    ext_syn = StaticSynapse(weight=JE, delay=dt)

    print("%d Connecting excitatory population with connection probability %g, weight %g nA and delay %g ms." % (rank, epsilon, JE, delay))
    E_to_E = Projection(E_net, E_net, connector, E_syn, receptor_type="excitatory")
    print("E --> E\t\t", len(E_to_E), "connections")
    I_to_E = Projection(I_net, E_net, connector, I_syn, receptor_type="inhibitory")
    print("I --> E\t\t", len(I_to_E), "connections")
    input_to_E = Projection(expoisson, E_net, ext_Connector, ext_syn, receptor_type="excitatory")
    print("input --> E\t", len(input_to_E), "connections")

    print("%d Connecting inhibitory population with connection probability %g, weight %g nA and delay %g ms." % (rank, epsilon, JI, delay))
    E_to_I = Projection(E_net, I_net, connector, E_syn, receptor_type="excitatory")
    print("E --> I\t\t", len(E_to_I), "connections")
    I_to_I = Projection(I_net, I_net, connector, I_syn, receptor_type="inhibitory")
    print("I --> I\t\t", len(I_to_I), "connections")
    input_to_I = Projection(inpoisson, I_net, ext_Connector, ext_syn, receptor_type="excitatory")
    print("input --> I\t", len(input_to_I), "connections")

    # read out time used for building
    buildCPUTime = timer.elapsedTime()
    # === Run simulation ===========================================================

    # run, measure computer time
    timer.start()  # start timer on construction
    print("%d Running simulation for %g ms (dt=%sms)." % (rank, simtime, dt))
    run(simtime)
    print("Done")
    simCPUTime = timer.elapsedTime()

    # write data to file
    #print("%d Writing data to file." % rank)
    #(E_net + I_net).write_data("Results/brunel_np%d_%s.pkl" % (np, simulator_name))
    if save and not simulator_name=='neuroml':
        for pop in [E_net , I_net]:
            io = PyNNTextIO(filename="brunel-PyNN-%s-%s-%i.gdf"%(simulator_name, pop.label, rank))
            spikes =  pop.get_data('spikes', gather=False)
            for segment in spikes.segments:
                io.write_segment(segment)
                
            io = PyNNTextIO(filename="brunel-PyNN-%s-%s-%i.dat"%(simulator_name, pop.label, rank))
            vs =  pop.get_data('v', gather=False)
            for segment in vs.segments:
                io.write_segment(segment)
            
    spike_data = {}
    spike_data['senders'] = []
    spike_data['times'] = []
    index_offset = 1
    for pop in [E_net , I_net]:
        if rank == 0:
            spikes =  pop.get_data('spikes', gather=False)
            #print(spikes.segments[0].all_data)
            num_rec = len(spikes.segments[0].spiketrains)
            print("Extracting spike info (%i) for %i cells in %s"%(num_rec,pop.size,pop.label))
            #assert(num_rec==len(spikes.segments[0].spiketrains))
            for i in range(num_rec):
                ss = spikes.segments[0].spiketrains[i]
                for s in ss:
                    index = i+index_offset
                    #print("Adding spike at %s in %s[%i] (cell %i)"%(s,pop.label,i,index))
                    spike_data['senders'].append(index)
                    spike_data['times'].append(s)
            index_offset+=pop.size
        
    #from IPython.core.debugger import Tracer
    #Tracer()()

    E_rate = E_net.mean_spike_count()*1000.0/simtime
    I_rate = I_net.mean_spike_count()*1000.0/simtime

    # write a short report
    nprint("\n--- Brunel Network Simulation ---")
    nprint("Nodes              : %d" % np)
    nprint("Number of Neurons  : %d" % N)
    nprint("Number of Synapses : %d" % Nsyn)
    nprint("Input firing rate  : %g" % p_rate)
    nprint("Excitatory weight  : %g" % JE)
    nprint("Inhibitory weight  : %g" % JI)
    nprint("Excitatory rate    : %g Hz" % E_rate)
    nprint("Inhibitory rate    : %g Hz" % I_rate)
    nprint("Build time         : %g s" % buildCPUTime)
    nprint("Simulation time    : %g s" % simCPUTime)

    # === Clean up and quit ========================================================

    end()
    
    
    if simulator_name=='neuroml' and jnml_simulator:
        from pyneuroml import pynml
        lems_file = 'LEMS_Sim_PyNN_NeuroML2_Export.xml'
        
        print('Going to run generated LEMS file: %s on simulator: %s'%(lems_file,jnml_simulator))
        
        if jnml_simulator=='jNeuroML':
            results, events = pynml.run_lems_with_jneuroml(lems_file, nogui=True, load_saved_data=True, reload_events=True)
        
        elif jnml_simulator=='jNeuroML_NEURON':
            results, events = pynml.run_lems_with_jneuroml_neuron(lems_file, nogui=True, load_saved_data=True, reload_events=True)
            
        spike_data['senders'] = []
        spike_data['times'] = []
        for k in events.keys():
            values = k.split('/') 
            index = int(values[1]) if values[0]=='E_net' else NE+int(values[1])
            n = len(events[k])
            print("Loading spikes for %s (index %i): [%s, ..., %s (n=%s)] sec"%(k,index,events[k][0] if n>0 else '-',events[k][-1] if n>0 else '-',n))
            for t in events[k]:
                spike_data['senders'].append(index)
                spike_data['times'].append(t*1000)
                
    #print spike_data
    return spike_data

if __name__ == '__main__':

    simulator_name = get_script_args(1)[0]
    simtime = 1000.0
    order = 100
    dt = 0.1
    jnml_simulator=None
    
    if simulator_name == 'neuroml':
        jnml_simulator='jNeuroML_NEURON'
        dt=0.025
        order=10
    
    eta         = 1.0     # rel rate of external input
    g           = 4.0

    runBrunelNetwork(g=g, 
                     eta=eta, 
                     simtime = simtime, 
                     dt = dt,
                     order = order, 
                     save=True, 
                     N_rec_v=20, 
                     simulator_name=simulator_name,
                     jnml_simulator=jnml_simulator,
                     epsilon = 0.1)