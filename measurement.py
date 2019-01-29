from experiment import Experiment
from controller_and_memory import SetController
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


def controller_planet_embedding_correlation(model_name, model_folders, n_measurements, name=None):
    experiment = Experiment.load(model_name, model_folders)
    assert isinstance(experiment.agent.controller_and_memory.controller, SetController)

    experiment.conf.max_imaginations_per_action = 0

    # experiment.agent.controller_and_memory.controller.measure_effect_embeddings = True
    # experiment.agent.controller_and_memory.controller.embeddings = []
    # experiment.agent.controller_and_memory.controller.metrics = []

    experiment.agent.imaginator.measure_effect_embeddings = True
    experiment.agent.imaginator.embeddings = []
    experiment.agent.imaginator.metrics = []

    experiment.evaluate(n_measurements)

    # embedding_matrix = np.stack(experiment.agent.controller_and_memory.controller.embeddings)
    embedding_matrix = np.stack(experiment.agent.imaginator.embeddings)

    pca = PCA(n_components=3)
    pca.fit(embedding_matrix)
    pca_results = pd.DataFrame(pca.transform(embedding_matrix), columns=['pc1', 'pc2', 'pc3'])

    print(np.var(embedding_matrix, axis=0))
    print(embedding_matrix)
    # print(experiment.agent.controller_and_memory.controller.embeddings)

    metrics = pd.DataFrame(
        # experiment.agent.controller_and_memory.controller.metrics,
        experiment.agent.imaginator.metrics,
        columns=['mass', 'dist', 'force']
    )

    results = pd.concat([pca_results, metrics], axis=1)

    for pc in ['pc1', 'pc2', 'pc3']:
        scatter_plot = results.plot.scatter(
            x='force',
            y=pc,
            c='dist',
            s=results['mass'] * 250,
            colormap='viridis',
            # loglog=True
        ).get_figure()

        scatter_plot.savefig('imag_image_{}.png'.format(pc))

    import matplotlib.pyplot as plt
    pd.plotting.scatter_matrix(results)
    plt.savefig('imag_scatter_matrix.png')


controller_planet_embedding_correlation("set_controller_bugtest-4p-4imag-nofuel", ('storage', 'home', 'misc'), 100)
