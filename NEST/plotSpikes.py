import sys
from pyneuroml import pynml

xs = []
ys = []
labels = []
markers = []
linestyles = []

max_id = 0

for file_name in sys.argv[1:]:
    print("Loading spike times from: %s"%file_name)
    spikes_file = open(file_name)
    x = []
    y = []
    max_id_here = 0
    for line in spikes_file:
        [id, t] = line.split()
        id_shift = max_id+int(id)
        max_id_here = max(max_id_here,id_shift) 
        x.append(t)
        y.append(id_shift)
    max_id = max_id_here
    xs.append(x)
    ys.append(y)
    labels.append(spikes_file.name)
    markers.append('.')
    linestyles.append('')



pynml.generate_plot(xs,
                    ys, 
                    "Spike times from: %s"%spikes_file.name, 
                    labels = labels, 
                    linestyles=linestyles,
                    markers=markers,
                    xaxis = "Time (s)", 
                    yaxis = "Cell number", 
                    grid = False,
                    show_plot_already=True)
    