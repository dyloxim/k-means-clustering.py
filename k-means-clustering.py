# main.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def squared_dist(u, v):
    running_total = 0
    for i in range(len(u)):
        running_total += (u[i] - v[i])**2
    return running_total


def by_components(vecs):
    indicies = range(0, len(vecs[0]))
    components = [np.array([]) for i in indicies]
    for i in indicies:
        components[i] = [vec[i] for vec in vecs]
    return components


def cluster(i):
    global state
    return [] if state['clusterings'] is None else [state['coords'][coord_idx]
          for coord_idx, cluster_idx in enumerate(state['clusterings'])
          if cluster_idx == i]


def update_clusterings():
    global state
    state['clusterings'] = np.empty(len(state['coords']), dtype=int)
    for i, coord in enumerate(state['coords']):
        squared_dists = [squared_dist(coord, centroid)
                         for centroid in state['centroids']]
        closest_cluster = np.where(
            squared_dists == np.min(squared_dists))[0][0]
        state['clusterings'][i] = closest_cluster
    state['clusterings'] = np.array(state['clusterings'], dtype=int)


def update_centroids():
    global state
    state['centroids'] = [centroid_of(cluster(i))
                          for i, _ in enumerate(state['centroids'])]


def centroid_of(coords):
    N = len(coords[0])
    centroid = np.empty(N)
    for i, _ in enumerate(coords[0]):
        centroid[i] = np.average([x[i] for x in coords])
    return centroid


def init_state():
    global state 
    state = {
        'coords': np.array(np.random.rand(1500,2), dtype=np.float64),
        'centroids': np.array(np.random.rand(15,2), dtype=np.float64),
        'clusterings': None
    }
    state['clusterings'] = [np.array([], dtype=int) for i in range(state_meta('k'))]


def state_meta(key):
    global state
    return {
        'k': len(state['centroids']),
        'clusters': [cluster(i) for i in range(len(state['centroids']))]
    }[key]


def animate(i):
    global state, ax, subplots
    ax.set_xlabel("Frame " + str(i))
    if i == 0:
        init_state()
        init_fig()
    else:
        next_state()
        update_subplots()


def update_subplots():
    global state, subplots
    subplots['coords'].set_offsets(state['coords'])
    for i, cluster_plot in enumerate(subplots['clusters']):
        if state_meta('clusters')[i] != []:
            cluster_plot.set_offsets(state_meta('clusters')[i])
    subplots['centroids'].set_offsets(state['centroids'])


def init_fig():
    global ax, subplots
    ax.clear()
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    subplots = {
        'coords': ax.scatter([], [], zorder=1),
        'clusters': [ax.scatter([], [], zorder=1) for i in range(state_meta('k'))],
        'centroids': ax.scatter([], [], marker='x', c='black', zorder=2),
    }
    ax.set_xlabel("Frame 0")
    update_subplots()

state = None

def run():
    global state, ax, subplots
    init_state()
    fig = plt.figure()
    ax = fig.add_subplot(xlim=(-1, 4), ylim=(-1, 4))
    ani = FuncAnimation(fig, animate, frames=range(6), repeat=True, interval=800)
    plt.show()


def next_state():
    global state
    update_clusterings()
    update_centroids()


run()
