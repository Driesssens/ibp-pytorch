import pandas
import ast
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
from copy import copy
import numpy as np
from itertools import product
import plotly.offline as pyo
import plotly.plotly as py
import plotly.graph_objs as go
from ipywidgets import interactive, HBox, VBox
from collections import defaultdict
from utilities import get_color, color_string

STANDARD_METRICS = (
    'controller/mean', 'imaginator/mean', 'rp_controller/mean', 'rp_imaginator/mean', 'manager_mean_ship_p', 'episodes_per_minute', 'mean_f_planets_in_each_imagination', 'mean_n_planets_in_each_imagination', 'imaginator_mean_dynamic_final_position_error',
    'imaginator_mean_static_final_position_error')

VALIDATED_METRICS = ('controller/mean', 'imaginator/mean', 'manager_mean_ship_p', 'mean_f_planets_in_each_imagination', 'mean_n_planets_in_each_imagination', 'imaginator_mean_dynamic_final_position_error', 'imaginator_mean_static_final_position_error')

STANDARD_TIME_AND_STEPS_BASED_ON = 'controller/mean'
CSVS_FOLDER = 'csvs'


def run_to_data_frame(path, metrics=STANDARD_METRICS, time_and_steps_based_on=STANDARD_TIME_AND_STEPS_BASED_ON, verbose=False, add_performance_agg=('all', 10000, 20000), add_manager_ship_p=True):
    tf_data = EventAccumulator(str(path), purge_orphaned_data=False).Reload().scalars

    if len(tf_data.Keys()) == 0:
        print("Skipping {} because it was crashed.".format(str(path)))
        return

    true_metrics = []
    filtered_tf_data = []

    for metric in metrics:
        try:
            data = tf_data.Items(metric)
            filtered_tf_data.append(data)
            true_metrics.append(metric)
        except KeyError:
            if verbose:
                print("No metric {} for {}".format(metric, path))
            continue

    # Get and validate all steps per key
    all_steps_per_key = [tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in (filtered_tf_data[:-1] if 'imaginator_mean_dynamic_final_position_error' in true_metrics else filtered_tf_data)]
    n_steps_per_key = [len(tup) for tup in all_steps_per_key]

    set_of_n_steps = set(n_steps_per_key)
    min_n_steps = min(set_of_n_steps)

    skipping_last_events = len(set_of_n_steps) == 2

    if len(set_of_n_steps) > 2 or \
            len(set(all_steps_per_key)) > 3 or \
            skipping_last_events and max(set_of_n_steps) - min_n_steps != 1:
        print("Skipping {} because not all metrics had same (number of) steps.".format(str(path)))
        print(set(n_steps_per_key))
        print(set(all_steps_per_key))
        print("")
        return

    if skipping_last_events:
        print("Some last events of step #{} will be skipped for {}".format(min_n_steps + 1, str(path)))

    steps = []
    times = []
    valuess = {}
    imaginator_mean_dynamic_final_position_error_steps = []
    imaginator_mean_dynamic_final_position_error_len = None

    for metric, tf_events in zip(true_metrics, filtered_tf_data):
        values = []

        if time_and_steps_based_on == metric:
            for i_step, tf_event in enumerate(tf_events):
                if i_step < min_n_steps:
                    steps.append(tf_event.step)
                    times.append(tf_event.wall_time)
                    values.append(tf_event.value)
        elif metric == 'imaginator_mean_dynamic_final_position_error':
            for i_step, tf_event in enumerate(tf_events):
                imaginator_mean_dynamic_final_position_error_steps.append(tf_event.step)
                values.append(tf_event.value)

            imaginator_mean_dynamic_final_position_error_len = len(values)
        elif metric == 'imaginator_mean_static_final_position_error':
            j_step = 0

            for i_step, tf_event in enumerate(tf_events):
                step = tf_event.step

                while step != imaginator_mean_dynamic_final_position_error_steps[j_step]:
                    values.append(None)
                    j_step += 1

                values.append(tf_event.value)
                j_step += 1

            while len(values) != imaginator_mean_dynamic_final_position_error_len:
                values.append(None)

        else:
            values = [tf_event.value for tf_event in tf_events]

        valuess[metric] = values[:min_n_steps]

    valuess['times'] = [time - times[0] for time in times]

    if 'manager_mean_ship_p' not in true_metrics and add_manager_ship_p:
        valuess['manager_mean_ship_p'] = [1.0 for _ in steps]
        true_metrics.append('manager_mean_ship_p')

        if verbose:
            print("Added manager_mean_ship_p = 1.0 to {}".format(path))

    data_frame = pandas.DataFrame(valuess, index=steps, columns=['times'] + true_metrics)

    for window in ('all', 10000, 20000):
        min_indices = []

        for i_end in data_frame.index:
            i_start = max(0, i_end - window) if window != 'all' else 0
            min_indices.append(data_frame['rp_controller/mean'].loc[i_start: i_end].idxmin())

        data_frame = data_frame.assign(**{"{}_{}".format(window, metric): data_frame[metric].loc[min_indices].values for metric in VALIDATED_METRICS if metric in data_frame.columns})

    return data_frame


def runs_to_csvs(path, metrics=STANDARD_METRICS, time_and_steps_based_on=STANDARD_TIME_AND_STEPS_BASED_ON, verbose=False, exclude=()):
    (Path(CSVS_FOLDER) / path).mkdir(parents=True, exist_ok=True)

    for experiment_path in path.glob('*'):
        if experiment_path.name in exclude:
            print("Excluded {}".format(experiment_path))
            continue

        df = run_to_data_frame(experiment_path, metrics, time_and_steps_based_on, verbose=verbose)

        if df is not None:
            print("Storing run {} as csv".format(experiment_path))
            df.to_csv(Path(CSVS_FOLDER) / experiment_path)


class Runs:
    def __init__(self, folder_path, done_hours, done_steps):
        csvs_path = Path(CSVS_FOLDER) / folder_path
        self.runs = [Run(experiment_path, done_hours=done_hours, done_steps=done_steps) for experiment_path in csvs_path.glob('*')]
        self.conf = defaultdict(set)
        self.folder = folder_path.name
        self.selection = 'all'

        for run in self.runs:
            for setting, value in run.conf.items():
                self.conf[setting].add(value)

    def __repr__(self):
        return "Runs({}, {}, {})".format(len(self), self.folder, self.conf.keys())

    def id(self, queried_id):
        return [run for run in self.runs if run.id == queried_id][0]

    def __call__(self, **kwargs):
        new = copy(self)
        new.runs = [run for run in self.runs if run(**kwargs)]
        new.selection = '-'.join('{}_{}'.format(left, right) for left, right in kwargs.items())
        return new

    def group(self, group_settings):
        groups = []
        remaining_confs = {key: value for key, value in self.conf.items() if not (key in group_settings or key == 'id')}

        for conf in dict_product(remaining_confs):
            runs = self(**conf).runs

            if len(runs) == 0:
                continue

            for setting in group_settings:
                conf[setting] = 'grouped'

            print(runs)
            groups.append(Group(runs=runs, conf=conf))

        return groups

    def only_done(self):
        new = copy(self)
        new.runs = [run for run in self.runs if run.done]
        return new

    def only_not_done(self):
        new = copy(self)
        new.runs = [run for run in self.runs if not run.done]
        return new

    def __getitem__(self, key):
        return self.id(key)

    def __len__(self):
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def controller(self, last_n_steps=None, until_step=None, pooling='validation', n_ci_samples=None, ci=(0.05, 0.95)):
        return self.metric('controller/mean', last_n_steps=last_n_steps, until_step=until_step, pooling=pooling, n_ci_samples=n_ci_samples, ci=ci)

    def metric(self, metric_name, last_n_steps=None, until_step=None, pooling='validation', n_ci_samples=None, ci=(0.05, 0.95)):
        metrics = [run.metric(metric_name=metric_name, last_n_steps=last_n_steps, until_step=until_step, pooling=pooling) for run in self]
        mean = sum(metrics) / len(metrics)

        if n_ci_samples is None:
            return mean
        else:
            return mean, bootstrap_ci(metrics, n_samples=n_ci_samples, ci=ci)

    def compare(self, compare_setting, group_settings=(), metric_name='controller/mean', last_n_steps=None, until_step=None, pooling='validation', n_ci_samples=10000, ci=(0.05, 0.95)):
        group_settings = group_settings + ('v', 'id')
        antagonists = [self(**{compare_setting: value}) for value in self.conf[compare_setting]]

        relevant_conf = {key: value for key, value in self.conf.items() if (key != compare_setting and key not in group_settings)}

        for conf in dict_product(relevant_conf):
            print("For {}:".format(conf))
            for antagonist in [antagonist(**conf) for antagonist in antagonists]:
                if len(antagonist) == 0:
                    print("\t<empty>")
                else:
                    print("\t{}: \t{}\t#{}".format(
                        antagonist.runs[0].conf[compare_setting],
                        nice(antagonist.metric(metric_name=metric_name, last_n_steps=last_n_steps, until_step=until_step, pooling=pooling, n_ci_samples=n_ci_samples, ci=ci)),
                        len(antagonist)
                    ))
            print("")
        print("")


class Group:
    def __init__(self, runs, conf, time=False):
        self.runs = runs
        self.conf = conf
        self.time = time

        self.up = None
        self.low = None
        self.df = None
        self.n = None
        self.n_changes = None

        self.aggregate()

    @property
    def steps(self):
        return not self.time

    def aggregate(self):
        if self.time:
            self.aggregate_time()
        else:
            self.aggregate_steps()

    def aggregate_time(self):
        pass

    def dict(self, n):
        d = {'c': n, 'n': len(self)}
        for setting, value in self.conf.items():
            if value != 'grouped':
                d[setting] = str(value)

        return d

    def aggregate_steps(self):
        index = self.runs[0].df.index

        for run in self.runs[1:]:
            index = index.union(run.df.index)

        columns = set.union(*[set(run.df.columns.values.tolist()) for run in self.runs])

        amounts = []
        means = defaultdict(list)
        upper_bounds = defaultdict(list)
        lower_bounds = defaultdict(list)

        for i in index:
            runs_here = [run.df.loc[i] for run in self.runs if i in run.df.index]
            amounts.append(len(runs_here))

            for column in columns:
                samples = [run[column] for run in runs_here]
                mean = sum(samples) / len(samples)
                low, up = bootstrap_ci(samples, 2000)

                means[column].append(mean)
                lower_bounds[column].append(low)
                upper_bounds[column].append(up)

        self.n = pandas.Series(data=amounts, index=index)
        self.up = pandas.DataFrame(data=upper_bounds, index=index)
        self.low = pandas.DataFrame(data=lower_bounds, index=index)
        self.df = pandas.DataFrame(data=means, index=index)
        self.n_changes = sorted([run.df.index.max() for run in self.runs])

    def __repr__(self):
        return "Group({} [{}])".format(self.name, len(self))

    def __len__(self):
        return len(self.runs)

    @property
    def name(self):
        return '-'.join('{}_{}'.format(left, right) for left, right in self.conf.items() if right is not 'grouped')

    def trace(self, column, color, column2=None, sec=False, hours=False):
        thing = [
            go.Scatter(
                name='0.05',
                x=self.df.index,
                y=self.low[column],
                marker=go.scatter.Marker(color=color_string(color, alpha=0.1)),
                line=dict(width=0),
                mode='lines',
                showlegend=False,
                hoverinfo='skip'),
            go.Scatter(
                name=self.name,
                x=self.df.index,
                y=self.df[column],
                text=self.n,
                hovertemplate="%{y:.4f} [%{text}]",
                mode='lines',
                line=go.scatter.Line(color=color_string(color)),
                fillcolor=color_string(color, alpha=0.1),
                fill='tonexty'),
            go.Scatter(
                name='0.95',
                x=self.df.index,
                y=self.up[column],
                mode='lines',
                marker=go.scatter.Marker(color=color_string(color, alpha=0.1)),
                line=dict(width=0),
                fillcolor=color_string(color, alpha=0.1),
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip'),
            go.Scatter(
                x=self.n_changes,
                y=[self.df[column][step] for step in self.n_changes],
                mode='markers',
                hoverinfo='text',
                text=list(reversed(range(len(self.n_changes)))),
                textposition='bottom center',
                showlegend=False,
                marker=go.scatter.Marker(color=color_string(color), symbol='diamond')
            )
        ]

        if column2 is not None:
            thing.append(
                go.Scatter(
                    name=self.name,
                    x=self.df.index,
                    y=self.df[column2],
                    text=self.n,
                    hovertemplate="%{y:.4f} [%{text}]",
                    mode='lines',
                    showlegend=False,
                    line=go.scatter.Line(color=color_string(color), dash='solid' if sec else 'dot'),
                    yaxis='y2'),
            )

        return thing


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))


def bootstrap_ci(original, n_samples, ci=(0.05, 0.95)):
    return np.quantile(np.random.choice(np.array(original), replace=True, size=(len(original), n_samples)).mean(axis=0), ci)


def nice(number):
    if not isinstance(number, tuple):
        return "{:0.3f}".format(number)
    else:
        return "{:0.3f} [{:0.3f}, {:0.3f}]".format(number[0], number[1][0], number[1][1])


def printn(number):
    print(nice(number))


class Run:
    def __init__(self, experiment_path, done_hours=None, done_steps=None):
        self.name = experiment_path.name
        self.conf = configuration_from_experiment_name(self.name)
        self.df = pandas.read_csv(experiment_path, index_col=0)  # type: pandas.DataFrame

        self.done = None

        if done_hours is not None:
            self.done = self.df.times.iat[-1] / 60 / 60 > done_hours

        if done_steps is not None:
            self.done = self.done or (self.df.index[-1] >= done_steps)

    def metric(self, metric_name, last_n_steps=None, until_step=None, pooling='validation'):
        data = self if until_step is None else self[:until_step]

        if last_n_steps is None:
            return data.df[metric_name].iat[-1]
        else:
            data = data[-last_n_steps:]

            if pooling == 'validation':
                max_rows = data.df[data.df[metric_name] == data.df[metric_name].min()]
                max_validations = max_rows['rp_' + metric_name]
                latest_max_validation = max_validations.iat[-1]
                return latest_max_validation
            elif pooling == 'best':
                return data.df[metric_name].min()
            elif pooling == 'mean':
                return data.df[metric_name].mean()

    def controller(self, last_n_steps=None, until_step=None, pooling='validation'):
        return self.metric('controller/mean', last_n_steps=last_n_steps, until_step=until_step, pooling=pooling)

    def __repr__(self):
        return "Run(done={}, {})".format(self.done, self.name)

    @property
    def id(self):
        return self.conf['id']

    def __call__(self, **kwargs):
        return all(self.conf[setting] == value for setting, value in kwargs.items())

    def __getitem__(self, key):
        if isinstance(key, slice):
            new = copy(self)

            from_step = (self.df.index.max() + key.start if key.start is not None and key.start < 0 else key.start)
            to_step = (self.df.index.max() + key.stop if key.stop is not None and key.stop < 0 else key.stop)

            if from_step is None:
                new.df = self.df[self.df.index <= to_step]
            elif to_step is None:
                new.df = self.df[self.df.index >= from_step]
            else:
                new.df = self.df[(self.df.index >= from_step) & (self.df.index <= to_step)]

            return new

        elif isinstance(key, int):
            key = (self.df.index.max() + key if key < 0 else key)
            return self.df.loc[key]
        else:
            raise TypeError


def configuration_from_experiment_name(experiment_name):
    configuration = {}

    for substring in experiment_name.split('-'):
        setting, _, value = substring.rpartition('_')

        configuration[setting] = parse_value(value)

    return configuration


def parse_value(value):
    if value == 'True':
        return True
    if value == 'False':
        return False
    if value == 'None':
        return None

    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    return value


def test():
    path = Path() / 'storage' / 'lisa3' / 'formal1'
    runs = Runs(path, done_hours=20, done_steps=100000)
    run = runs[3]

    for window in ('all', 10000, 20000):
        min_indices = []

        for i_end in run.df.index:
            i_start = max(0, i_end - window) if window != 'all' else 0
            min_indices.append(run.df['rp_controller/mean'].loc[i_start: i_end].idxmin())

        run.df = run.df.assign(**{"{}_{}".format(window, metric): run.df[metric].loc[min_indices].values for metric in VALIDATED_METRICS if metric in run.df.columns})

    datas = [go.Scatter(x=run.df.index, y=run.df[col], mode='lines', name=col) for col in run.df.columns]

    pyo.plot(
        figure_or_data=go.Figure(
            data=datas,
            layout=go.Layout(yaxis=dict(type='log', autorange=True))
        ),
        auto_open=True
    )

    return

    # return runs
    data_1 = go.Scatter(x=run.df.index, y=run.df['controller/mean'], mode='lines', name='controller/mean')
    data_2 = go.Scatter(x=run.df.index, y=run.df['controller/mean'].cummin(), mode='lines', name='cummin')
    data_3 = go.Scatter(x=run.df.index, y=run.df['rp_controller/mean'], mode='lines', name='rp_controller/mean')

    data_n = []
    for window in ('all', 10000, 20000):
        values = []

        for i_end in run.df.index:
            i_start = max(0, i_end - window) if window != 'all' else 0
            values.append(run.df['rp_controller/mean'].loc[run.df['controller/mean'].loc[i_start: i_end].idxmin()])
        data_n.append(go.Scatter(x=run.df.index, y=values, mode='lines', name='performance_{}'.format(window)))

    pyo.plot(
        figure_or_data=go.Figure(
            data=[data_1, data_2, data_3] + data_n,
            layout=go.Layout(yaxis=dict(type='log', autorange=True))
        ),
        auto_open=True
    )


def test2():
    # https://plot.ly/python/dropdowns/

    pyo.init_notebook_mode()

    path = Path() / 'storage' / 'lisa3' / 'formal1'
    runs = Runs(path, done_hours=20, done_steps=100000)
    run = runs[3]

    f = go.FigureWidget([go.Scatter(y=run.df['controller/mean'], x=run.df.index, mode='lines')])
    scatter = f.data[0]
    N = len(run.df)
    scatter.x = scatter.x + np.random.rand(N) / 10 * (run.df['controller/mean'].max() - run.df['controller/mean'].min())
    scatter.y = scatter.y + np.random.rand(N) / 10 * (run.df['controller/mean'].max() - run.df['controller/mean'].min())

    def update_axes(xaxis, yaxis):
        scatter = f.data[0]
        scatter.x = run.df[xaxis]
        scatter.y = run.df[yaxis]
        with f.batch_update():
            f.layout.xaxis.title = xaxis
            f.layout.yaxis.title = yaxis
            scatter.x = scatter.x + np.random.rand(N) / 10 * (run.df[xaxis].max() - run.df[xaxis].min())
            scatter.y = scatter.y + np.random.rand(N) / 10 * (run.df[yaxis].max() - run.df[yaxis].min())

    axis_dropdowns = interactive(update_axes, yaxis=run.df.select_dtypes('int64').columns, xaxis=run.df.select_dtypes('int64').columns)

    # Put everything together

    py.iplot(dict(data=VBox((HBox(axis_dropdowns.children), f))))


def test3():
    path = Path() / 'storage' / 'lisa3' / 'formal1'
    runs = Runs(path, done_hours=20, done_steps=100000)
    run = runs[0]

    trace = go.Scatter(
        x=run.df.index, y=run.df['controller/mean'],  # Data
        mode='lines', name='controller/mean'  # Additional options
    )

    trace2 = go.Scatter(
        x=run.df.index, y=run.df['imaginator/mean'],  # Data
        mode='lines', name='imaginator/mean'  # Additional options
    )

    def getDataByButton(df, trace):
        # return arg list to set x, y and chart title
        return [dict(data=trace)]

    updatemenus = list([
        dict(
            buttons=list([
                dict(label='controller/mean',
                     method='update',
                     args=getDataByButton(run.df, trace)
                     ),
                dict(label='imaginator/mean',
                     method='update',
                     args=getDataByButton(run.df, trace2)
                     ),
            ]),
            direction='left',
            pad={'r': 10, 't': 10},
            showactive=True,
            type='buttons',
            x=0.1,
            xanchor='left',
            y=1.1,
            yanchor='top'
        )
    ])

    layout = go.Layout(
        title='controller/mean',
        showlegend=True,
        width=1200,
        height=600,
        font=dict(
            family='Calibri, sans-serif',
            size=18,
            color='rgb(0,0,0)'),

        xaxis=dict(
            linewidth=1,
            title='steps',
            tickangle=-45,
            tickfont=dict(
                size=12
            )
        ),
        yaxis=dict(
            linewidth=1,
            tickfont=dict(
                size=12
            )
        ),
        updatemenus=updatemenus
    )

    fig = go.Figure(data=[trace], layout=layout)
    pyo.plot(fig)

# runs_to_csvs(Path() / 'storage' / 'final' / 'formal1')
# runs_to_csvs(Path() / 'storage' / 'final' / 'formal2')
