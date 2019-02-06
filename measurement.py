from experiment import Experiment
from controller_and_memory import SetController
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os
import datetime


def imaginator_planet_embedding_introspection(model_name, model_folders, n_measurements, name=None):
    experiment = Experiment.load(model_name, model_folders)

    if name is None:
        name = "{}_{}".format(model_name, datetime.datetime.now().strftime("%Y-%m-%d-%H.%M"))

    folders = ['measurements', 'imaginator_planet_embedding_introspection', name]
    os.makedirs(os.path.join(*folders))

    experiment.conf.max_imaginations_per_action = 0

    experiment.agent.imaginator.measure_imaginator_planet_embedding_introspection = True
    experiment.agent.imaginator.embeddings = []
    experiment.agent.imaginator.metrics = []

    experiment.evaluate(n_measurements)

    embedding_statistics = analyze_embeddings(experiment.agent.imaginator.embeddings, 3, True)

    metrics = pd.DataFrame(
        experiment.agent.imaginator.metrics,
        columns=['mass', 'dist', 'force']
    )

    results = pd.concat([embedding_statistics, metrics], axis=1)

    for metric in ['norm', 'pc1', 'pc2', 'pc3']:
        scatter_plot = results.plot.scatter(
            x='force',
            y=metric,
            c='dist',
            s=(results['mass'] ** 2) * 250,
            colormap='viridis',
        ).get_figure()

        scatter_plot.savefig(os.path.join(*(folders + ['scatter_{}.png'.format(metric)])))

    import matplotlib.pyplot as plt
    pd.plotting.scatter_matrix(results)
    plt.savefig(os.path.join(*(folders + ['matrix_scatter'])))


def controller_planet_embedding_introspection(model_name, model_folders, n_measurements, name=None):
    experiment = Experiment.load(model_name, model_folders)
    assert isinstance(experiment.agent.controller_and_memory.controller, SetController)

    if name is None:
        name = "{}_{}".format(model_name, datetime.datetime.now().strftime("%Y-%m-%d-%H.%M"))

    folders = ['measurements', 'controller_planet_embedding_introspection', name]
    os.makedirs(os.path.join(*folders))

    experiment.conf.max_imaginations_per_action = 0

    experiment.agent.controller_and_memory.controller.measure_controller_planet_embedding_introspection = True
    experiment.agent.controller_and_memory.controller.embeddings = []
    experiment.agent.controller_and_memory.controller.metrics = []

    experiment.evaluate(n_measurements)

    embedding_statistics = analyze_embeddings(experiment.agent.controller_and_memory.controller.embeddings, 3, True)

    metrics = pd.DataFrame(
        experiment.agent.controller_and_memory.controller.metrics,
        columns=['mass', 'dist', 'force']
    )

    results = pd.concat([embedding_statistics, metrics], axis=1)

    for metric in ['norm', 'pc1', 'pc2', 'pc3']:
        scatter_plot = results.plot.scatter(
            x='force',
            y=metric,
            c='dist',
            s=(results['mass'] ** 2) * 250,
            colormap='viridis',
        ).get_figure()

        scatter_plot.savefig(os.path.join(*(folders + ['scatter_{}.png'.format(metric)])))

    import matplotlib.pyplot as plt
    pd.plotting.scatter_matrix(results)
    plt.savefig(os.path.join(*(folders + ['matrix_scatter'])))


def analyze_embeddings(embeddings, n_pc=3, print_variation=False):
    embedding_matrix = np.stack(embeddings)

    if print_variation:
        print(np.var(embedding_matrix, axis=0))

    norms = pd.DataFrame(
        np.linalg.norm(embedding_matrix, axis=1),
        columns=['norm']
    )

    pca = PCA(n_components=n_pc)
    pca.fit(embedding_matrix)
    pcas = pd.DataFrame(pca.transform(embedding_matrix), columns=['pc{}'.format(i + 1) for i in range(n_pc)])

    return pd.concat([norms, pcas], axis=1)


def history_embedding_introspection(model_name, model_folders, n_measurements, name=None):
    experiment = Experiment.load(model_name, model_folders)

    if name is None:
        name = "{}_{}".format(model_name, datetime.datetime.now().strftime("%Y-%m-%d-%H.%M"))

    folders = ['measurements', 'history_embedding_introspection', name]
    os.makedirs(os.path.join(*folders))

    experiment.agent.controller_and_memory.memory.measure_history_embedding_introspection = True
    experiment.agent.controller_and_memory.memory.embeddings = []
    experiment.agent.controller_and_memory.memory.n_imagination = []

    experiment.evaluate(n_measurements)

    embedding_statistics = analyze_embeddings(experiment.agent.controller_and_memory.memory.embeddings, 3, True)

    n_imag = pd.DataFrame(
        experiment.agent.controller_and_memory.memory.n_imagination,
        columns=['n_imag']
    )

    results = pd.concat([n_imag, embedding_statistics], axis=1)
    splitted_results = [value for (key, value) in sorted({k: v for k, v in results.groupby('n_imag')}.items())]

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    for metric in ['norm', 'pc1', 'pc2', 'pc3']:
        _, bins = np.histogram(results[metric].values, bins='auto')
        histograms = [np.histogram(split[metric].values, bins=bins)[0] for split in splitted_results]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(histograms)):
            ys = np.append(histograms[i], [0])
            ax.bar(bins, ys.ravel(), zs=int(i), zdir='x', alpha=0.8, align='edge', width=[bins[i + 1] - bins[i] for i in range(len(bins) - 1)] + [0])

        ax.set_xlabel('imag')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('metric')
        ax.yaxis.set_ticks(bins[::3])
        ax.set_zlabel('n')
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')
        plt.show()
        # plt.savefig(os.path.join(*(folders + ['{}_histogram.png'.format(metric)])))


# imaginator_planet_embedding_introspection("set_controller_bugtest-4p-4imag-nofuel", ('storage', 'home', 'misc'), 100)
# imaginator_planet_embedding_introspection("setcontroller_effects_not-only_history_no_state", ('storage', 'home', 'misc'), 1000)
controller_planet_embedding_introspection("setcontroller_reactive", ('storage', 'home', 'misc'), 300)