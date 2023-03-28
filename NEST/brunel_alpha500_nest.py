from brunel_alpha_nest import runBrunelNetwork

simtime = 1000
order = 100

eta         = 2.0     # rel rate of external input
g           = 5.0

runBrunelNetwork(g=g, eta=eta, simtime = simtime, order = order, save=True, N_rec=100)