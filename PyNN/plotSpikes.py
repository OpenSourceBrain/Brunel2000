import sys
from pyneuroml import pynml

xs = []
ys = []
labels = []
markers = []
linestyles = []

offset_id = 0

files = sys.argv[1:]
'''
format = 'id_t'

if '-t_id' in files:
    format = 't_id'
    files.remove('-t_id')'''
    
format = 't_id'

for file_name in files:
    print("Loading spike times from: %s"%file_name)
    spikes_file = open(file_name)
    x = []
    y = []
    max_id_here = 0
    for line in spikes_file:
        if not line.startswith('#'):
            if format == 'id_t':
                [id, t] = line.split()
            elif format == 't_id':
                [t, id] = line.split()
            id_shift = offset_id+int(float(id))
            max_id_here = max(max_id_here,id_shift) 
            x.append(t)
            y.append(id_shift)
    print("max_id_here in %s: %i"%(file_name,max_id_here))
    labels.append("%s (%i cells)"%(spikes_file.name,max_id_here-offset_id))
    offset_id = max_id_here+1
    xs.append(x)
    ys.append(y)
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
    