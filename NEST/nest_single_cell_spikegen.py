import nest
import nest.raster_plot
import nest.voltage_trace

nest.ResetKernel()

dt = 0.005    # the resolution in ms
simtime = 1000.0  # Simulation time in ms

tauSyn = 1.0  # synaptic time constant in ms
tauMem = 20.0  # time constant of membrane potential in ms
CMem = 250.0  # capacitance of membrane in in pF
theta = 1.0  # membrane threshold potential in mV
neuron_params = {"C_m":        CMem,
                 "tau_m":      tauMem,
                 "tau_syn_ex": tauSyn,
                 "tau_syn_in": tauSyn,
                 "t_ref":      2.0,
                 "E_L":        0.0,
                 "V_reset":    -1.0,
                 "V_m":        0.0,
                 "V_th":       theta}

nest.SetKernelStatus({"resolution": dt, "print_time": True,
	 "overwrite_files": True, "data_path": "."})

print("Building network")
nest.SetDefaults("iaf_psc_alpha", neuron_params)

nodes_ex = nest.Create("iaf_psc_alpha", 1)
espikes = nest.Create("spike_detector")
s = [100., 120., 126., 135.]
noise = nest.Create("spike_generator",
                    params={"spike_times": s})
voltmeter = nest.Create("voltmeter", params={"interval": 0.005,
                        "to_file": True, "label": "nest_v"})

nest.SetStatus(espikes, [{"label": "iaf_exc",
                          "withtime": True,
                          "withgid": True}])

print("Connecting devices")

nest.CopyModel("static_synapse", "excitatory", {
               "weight": 50., "delay": 1.0})
nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(nodes_ex[:], espikes, syn_spec="excitatory")
nest.Connect(voltmeter, nodes_ex)


print("Simulating")

nest.Simulate(simtime)
events_ex = nest.GetStatus(espikes, "n_events")[0]
rate_ex = events_ex / simtime * 1000.0

print("Brunel network simulation (Python)")
print("Excitatory rate   : %.2f Hz" % rate_ex)

nest.voltage_trace.from_device(voltmeter)
#nest.raster_plot.from_device(espikes, hist=True)
