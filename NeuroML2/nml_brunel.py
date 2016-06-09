from numpy import exp, random
from scipy.special import lambertw
from neuroml import NeuroMLDocument
from neuroml import IafRefCell
from neuroml import Network
from neuroml import AlphaCurrentSynapse
from neuroml import Population
from neuroml import Projection
from neuroml import ConnectionWD
from neuroml import SpikeArray
from neuroml import Spike
import neuroml.writers as writers


def create_nml():
    nml_doc = NeuroMLDocument(id="Brunel")
    iaf = create_iaf_cell()
    nml_doc.iaf_ref_cells.append(iaf)

    noise = create_inputs()
    nml_doc.spike_arrays.append(noise)
    
    tauSyn = 0.5 # ms

    alpha_syn_ex = create_alpha_syn('alpha_syn_ex', tauSyn, 0)
    nml_doc.alpha_current_synapses.append(alpha_syn_ex)

    alpha_syn_ih = create_alpha_syn('alpha_syn_in', tauSyn, 0)
    nml_doc.alpha_current_synapses.append(alpha_syn_ih)
    
    net, ex_ibase, in_ibase = create_network(iaf, tauSyn, alpha_syn_ex, alpha_syn_ih, noise)
    nml_doc.networks.append(net)
    
    alpha_syn_ex.ibase = ex_ibase
    alpha_syn_ih.ibase = in_ibase
    

    nml_file = './brunel2000_mini.net.nml'
    writers.NeuroMLWriter.write(nml_doc, nml_file)
#    from neuroml.utils import validate_neuroml2
#    validate_neuroml2(nml_file)


def create_network(cell, tauSyn, syn_ex, syn_ih, inputs):

    g = 5.0  # ratio inhibitory weight/excitatory weight
    eta = 2.0  # external rate relative to threshold rate
    epsilon = 0.1  # connection probability

    order = 10
    NE = 4*order  # number of excitatory neurons
    NI = 1*order  # number of inhibitory neurons
    N_neurons = NE+NI   # number of neurons in total

    CE = int(epsilon*NE)  # number of excitatory synapses per neuron
    CI = int(epsilon*NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI+CE)    # total number of synapses per neuron

    net = Network(id="net")

    nodes_ex = Population(id="nodes_ex",  component=cell.id, size=NE)
    nodes_inh = Population(id="nodes_inh", component=cell.id, size=NI)
    noise = Population(id="noise", component=inputs.id, size=1)
    net.populations.append(nodes_ex)
    net.populations.append(nodes_inh)
    net.populations.append(noise)

    g = 5.0  # ratio inhibitory weight/excitatory weight
    J = 0.1  # postsynaptic amplitude in mV
    J_unit = computePSPnorm(cell.tauMem, cell.CMem, tauSyn)
    J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
    J_in = -g * J_ex   # amplitude of inhibitory postsynaptic current

    ex_ibase = str(J_ex) + 'nA'
    in_ibase = str(J_in) + 'nA'
    delay = "1.5 ms"
    
    proj = Projection(id="Proj0", synapse=syn_ex.id,
                        presynaptic_population=noise.id, 
                        postsynaptic_population=nodes_ex.id)
                        
    net.projections.append(proj)
    
    proj.connection_wds.extend(
        all_to_all(noise, nodes_ex, syn_ex, 1, delay))

    #net.synaptic_current_weight_delays.extend(
    #    all_to_all(noise, nodes_inh, synapse, in_weight, delay))
    '''
    random.seed(1234)
    sources_ex = random.random_integers(1, NE, (N_neurons, CE))
    sources_in = random.random_integers(NE+1, N_neurons, (N_neurons, CI))

    ex_ex = sources_ex[:NE] - 1
    ex_in = sources_ex[NE:] - 1
    in_ex = sources_in[:NE] - NE - 1
    in_in = sources_in[NE:] - NE - 1
    
    net.synaptic_current_weight_delays.extend(
        connect_from_list(ex_ex, 'nodes_ex', 'nodes_ex', synapse, 1, delay))

    net.synaptic_current_weight_delays.extend(
        connect_from_list(ex_in, 'nodes_ex', 'nodes_inh', synapse, 1, delay))

    net.synaptic_current_weight_delays.extend(
        connect_from_list(in_ex, 'nodes_inh', 'nodes_ex', synapse, 1, delay))

    net.synaptic_current_weight_delays.extend(
        connect_from_list(in_in, 'nodes_inh', 'nodes_inh', synapse, 1, delay))'''

    return net, ex_ibase, in_ibase


def connect_from_list(from_list, from_name, to_name, synapse, weight, delay):
    '''
    >>> connect_from_list([[0 1 2],[3 4 5]], 'from', 'to', ...)
    from[0]->to[0], from[1]->to[0], from[2]->to[0]
    from[3]->to[1], from[4]->to[1], from[5]->to[1]
    '''
    for to_i, from_ii in enumerate(from_list):
        for from_i in from_ii:
            fr = "%s[%i]" % (from_name, from_i)
            to = "%s[%i]" % (to_name, to_i)
            yield synaptic_connection(fr, to, synapse.id, weight, delay)


def all_to_all(pre, post, synapse, weight, delay):
    from itertools import product
    count=-1
    for ipre, ipost in product(xrange(pre.size), xrange(post.size)):
        count+=1
        fr = "%s[%i]" % (pre.id, ipre)
        to = "%s[%i]" % (post.id, ipost)
        yield synaptic_connection(id=count,pre_cell_id=fr, post_cell_id=to,weight=weight, delay=delay)


def synaptic_connection(id,pre_cell_id, post_cell_id, weight, delay):
    return ConnectionWD(id=id,pre_cell_id=pre_cell_id, post_cell_id=post_cell_id, delay=delay,
                                      weight=weight)


def create_inputs():

    # eta     = 2.0  # external rate relative to threshold rate
    # nu_th  = (theta * CMem) / (J_ex*CE*numpy.exp(1)*tauMem*tauSyn)
    # nu_ex  = eta*nu_th
    # p_rate = 1000.0*nu_ex*CE

    # need a poisson gen with p_rate

    spks = SpikeArray(id="spks")
    for i, t in enumerate([100, 120, 126, 135]):
        spks.spikes.append(Spike(id="%d" % i, time="%d ms" % t))
    return spks


def create_alpha_syn(id, tauSyn, ibase):

    tau = str(tauSyn) + 'ms'
    alpha = AlphaCurrentSynapse(id=id, tau=tau, ibase=ibase)
    return alpha


def create_iaf_cell():

    theta = "20 mV"
    reset = "0 mV"
    E_L = "0 mV"
    t_ref = "2 ms"

    CMem = 250.0  # pF
    tauMem = 20   # ms
    g_leak = str(CMem/tauMem) + 'nS'
    cm = str(CMem) + 'pF'

    iaf = IafRefCell(id="iaf", C=cm, thresh=theta, reset=reset,
                     leak_conductance=g_leak, leak_reversal=E_L, refract=t_ref)
    iaf.CMem = CMem
    iaf.tauMem = tauMem  # used for psp size calc
    return iaf


def computePSPnorm(tauMem, CMem, tauSyn):
    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)
    t_max = 1.0 / b * (-lambertw(-exp(-1.0 / a) / a, k=-1).real - 1.0 / a)
    return exp(1.0) / (tauSyn * CMem * b) * ((exp(-t_max / tauMem) - exp(-t_max / tauSyn)) / b - t_max * exp(-t_max / tauSyn))

if __name__ == '__main__':
    create_nml()
