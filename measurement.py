from experiment import Experiment
from controller_and_memory import SetController, SetMemory
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os
import datetime
from spaceship_environment import cartesian2polar, polar2cartesian


def imaginator_planet_embedding_introspection_2(model_name, model_folders, n_measurements, name=None):
    experiment = Experiment.load(model_name, model_folders)

    if name is None:
        name = "{}_{}".format(model_name, datetime.datetime.now().strftime("%Y-%m-%d-%H.%M"))

    folders = ['measurements', 'imaginator_planet_embedding_introspection2', name]
    os.makedirs(os.path.join(*folders))

    experiment.conf.max_imaginations_per_action = 0

    embeddings_list = []
    metrics_list = []

    experiment.initialize_environment()
    experiment.env.render_after_each_step = False

    for _ in range(n_measurements):
        experiment.env.reset()
        effect_embeddings = experiment.agent.imaginator.embed(experiment.env.agent_ship, experiment.env.planets)

        for i, embedding in enumerate(effect_embeddings):
            planet = experiment.env.planets[i]

            actual_radius = np.linalg.norm(experiment.env.agent_ship.xy_position - planet.xy_position)
            pretended_radius = actual_radius

            pretended_xy_distance = planet.xy_position - experiment.env.agent_ship.xy_position
            minimal_radius = planet.mass

            if pretended_radius < minimal_radius:
                pretended_radius = minimal_radius
                actual_angle, actual_radius = cartesian2polar(pretended_xy_distance[0], pretended_xy_distance[1])
                pretended_xy_distance = np.array(polar2cartesian(actual_angle, pretended_radius))

            xy_gravitational_force = experiment.env.gravitational_constant * planet.mass * experiment.env.agent_ship.mass * pretended_xy_distance / pretended_radius ** 3
            gravitational_force_magnitude = np.linalg.norm(xy_gravitational_force)

            embeddings_list.append(embedding.detach().numpy())
            metrics_list.append([planet.mass, actual_radius, gravitational_force_magnitude])

    embedding_statistics = analyze_embeddings(embeddings_list, 3, True)

    metrics = pd.DataFrame(
        metrics_list,
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
    print(embeddings)
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


def setmemory_object_embedding_introspection(model_name, model_folders, n_measurements, name=None):
    experiment = Experiment.load(model_name, model_folders)
    assert isinstance(experiment.agent.controller_and_memory.memory, SetMemory)

    if name is None:
        name = experiment_name(model_folders, model_name)

    folders = ['measurements', 'setmemory_object_embedding_introspection', name]
    os.makedirs(os.path.join(*folders))

    experiment.conf.max_imaginations_per_action = 1

    experiment.agent.controller_and_memory.memory.measure_setmemory_object_embedding_introspection = True
    experiment.agent.controller_and_memory.memory.embeddings = []
    experiment.agent.controller_and_memory.memory.metrics = []

    experiment.evaluate(n_measurements)

    embedding_statistics = analyze_embeddings(experiment.agent.controller_and_memory.memory.embeddings, 3, True)

    metrics = pd.DataFrame(
        experiment.agent.controller_and_memory.memory.metrics,
        columns=['isplan', 'mass', 'radius', 'x', 'y']
    )

    results = pd.concat([embedding_statistics, metrics], axis=1)

    for metric in ['norm', 'pc1', 'pc2', 'pc3']:
        scatter_plot = results.plot.scatter(
            x='radius',
            y=metric,
            c='isplan',
            s=(results['mass'] ** 2) * 250,
            colormap='viridis',
            logy=metric == 'norm'
        ).get_figure()

        scatter_plot.savefig(os.path.join(*(folders + ['scatter_{}.png'.format(metric)])))

    import matplotlib.pyplot as plt
    pd.plotting.scatter_matrix(results)
    plt.savefig(os.path.join(*(folders + ['matrix_scatter'])))


def performance_under_more_and_unobserved_planets(model_name, model_folders, n_measurements, name=None, only_normal=False):
    if name is None:
        name = experiment_name(model_folders, model_name)

    folders = ['measurements', 'controller_performance' if only_normal else 'performance_under_more_and_unobserved_planets', name]
    os.makedirs(os.path.join(*folders))

    # for mode in ('normal', 'only_ship_observed', 'extra_planets', 'extra_planets_unobserved'):
    for mode in ('normal', 'only_ship_observed'):

        if only_normal and mode in ('only_ship_observed', 'extra_planets', 'extra_planets_unobserved'):
            continue

        experiment = Experiment.load(model_name, model_folders)
        assert isinstance(experiment.agent.controller_and_memory.memory, SetMemory)

        experiment.agent.measure_performance = True
        experiment.agent.controller_and_memory_mean_task_cost_measurements = []
        experiment.agent.imaginator_mean_final_position_error_measurements = []
        experiment.agent.manager_mean_task_cost_measurements = []

        experiment.agent.controller_and_memory.measure_performance_under_more_and_unobserved_planets = mode

        if mode == 'extra_planets' or mode == 'extra_planets_unobserved':
            experiment.conf.n_secondary_planets += 1

        experiment.evaluate(n_measurements)

        imaginator_performance = np.stack(experiment.agent.controller_and_memory_mean_task_cost_measurements)

        results = pd.Series(imaginator_performance)

        import matplotlib.pyplot as plt
        results.plot.hist(grid=True)
        plt.title('{}. #{}. mean: {}'.format(mode, n_measurements, imaginator_performance.mean()))
        plt.ylabel('controller mean task cost')

        plt.savefig(os.path.join(*(folders + ['controller_task_cost_on_{}'.format(mode)])))


def analyze_actions(model_name, model_folders, n_measurements, name=None):
    if name is None:
        name = experiment_name(model_folders, model_name)

    folders = ['measurements', 'analyze_actions', name]
    os.makedirs(os.path.join(*folders))

    experiment = Experiment.load(model_name, model_folders)
    assert isinstance(experiment.agent.controller_and_memory.memory, SetMemory)

    experiment.agent.measure_analyze_actions = True
    experiment.agent.controller_and_memory.controller.all_actions = []
    experiment.agent.actual_actions = []

    experiment.evaluate(n_measurements)

    all_actions = pd.Series(np.stack(experiment.agent.controller_and_memory.controller.all_actions))
    actual_actions = pd.Series(np.stack(experiment.agent.actual_actions))

    import matplotlib.pyplot as plt
    all_actions.plot.hist(grid=True)
    plt.title('#{}. mean: {}. min: {}. max: {}.'.format(n_measurements, all_actions.mean(), all_actions.min(), all_actions.max()))
    plt.ylabel('controller all actions')
    plt.savefig(os.path.join(*(folders + ['controller_all_actions'])))

    actual_actions.plot.hist(grid=True)
    plt.title('#{}. mean: {}. min: {}. max: {}.'.format(n_measurements, actual_actions.mean(), actual_actions.min(), actual_actions.max()))
    plt.ylabel('controller actual actions')
    plt.savefig(os.path.join(*(folders + ['controller_actual_actions'])))


def experiment_name(model_folders, model_name, date_first=False):
    return "{}_{}.{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H.%M"), '.'.join(model_folders), model_name)


# imaginator_planet_embedding_introspection("set_controller_bugtest-4p-4imag-nofuel", ('storage', 'home', 'misc'), 100)
# imaginator_planet_embedding_introspection("setcontroller_effects_not-only_history_no_state", ('storage', 'home', 'misc'), 1000)
# controller_planet_embedding_introspection("setcontroller_reactive", ('storage', 'home', 'misc'), 300)
# setmemory_object_embedding_introspection("selu_False-embsize_50-use_i_imagination_False-use_action_True-hide_ship_state_True-agg_raw-id_5", ('storage', 'lisa', 'memoryless'), 500)

# performance_under_more_and_unobserved_planets("use_i_imagination_False-use_action_True-hide_ship_state_True-aggregate_layer_False-id_9", ('storage', 'lisa', 'prototype'), 500)
# performance_under_more_and_unobserved_planets("selu_False-use_action_True-v_3-id_7", ('storage', 'lisa', 'prototype2'), 1000, only_normal=True)
# performance_under_more_and_unobserved_planets("bugtest19-0.1-2", ('storage', 'home', 'ppo_bugtest'), 1000, only_normal=True)

# setmemory_object_embedding_introspection("selu_False-use_action_True-v_3-id_7", ('storage', 'lisa', 'prototype2'), 500)
# imaginator_planet_embedding_introspection_2("cur-scratch-2weightless-1", ('storage', 'home', 'imag'), 100)

# performance_under_more_and_unobserved_planets("selu_False-embsize_50-use_i_imagination_True-use_action_True-hide_ship_state_True-agg_raw-id_1", ('storage', 'lisa', 'memoryless'), 500)

# analyze_actions("v_7-memoryless_True-id_6", ('storage', 'lisa', 'varia_hleak'), 5000)

for i in range(5, 10):
    name = "v_{}-memoryless_True-id_{}".format(i + 1, i)
    performance_under_more_and_unobserved_planets(name, ('storage', 'lisa', 'varia_hleak'), 500)
