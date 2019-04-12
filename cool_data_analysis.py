import pandas
import ast
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
from copy import copy

STANDARD_METRICS = ('controller/mean', 'imaginator/mean', 'episodes_per_minute', 'manager_mean_ship_p')
STANDARD_TIME_AND_STEPS_BASED_ON = 'controller/mean'
CSVS_FOLDER = 'csvs'


def run_to_data_frame(path, metrics=STANDARD_METRICS, time_and_steps_based_on=STANDARD_TIME_AND_STEPS_BASED_ON):
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
            print("No metric {} for {}".format(metric, path))
            continue

    # Get and validate all steps per key
    all_steps_per_key = [tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in filtered_tf_data]
    n_steps_per_key = [len(tup) for tup in all_steps_per_key]

    set_of_n_steps = set(n_steps_per_key)
    min_n_steps = min(set_of_n_steps)

    skipping_last_events = len(set_of_n_steps) == 2

    if len(set_of_n_steps) > 2 or \
            len(set(all_steps_per_key)) > 2 or \
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

    for metric, tf_events in zip(true_metrics, filtered_tf_data):
        values = []

        if time_and_steps_based_on == metric:
            for tf_event in tf_events:
                steps.append(tf_event.step)
                times.append(tf_event.wall_time)
                values.append(tf_event.value)
        else:
            values = [tf_event.value for tf_event in tf_events][:min_n_steps]

        valuess[metric] = values

    valuess['times'] = [time - times[0] for time in times]

    data_frame = pandas.DataFrame(valuess, index=steps, columns=['times'] + true_metrics)
    return data_frame


def runs_to_csvs(path, metrics=STANDARD_METRICS, time_and_steps_based_on=STANDARD_TIME_AND_STEPS_BASED_ON):
    (Path(CSVS_FOLDER) / path).mkdir(parents=True, exist_ok=True)

    for experiment_path in path.glob('*'):
        df = run_to_data_frame(experiment_path, metrics, time_and_steps_based_on)

        if df is not None:
            print("Storing run {} as csv".format(experiment_path))
            df.to_csv(Path(CSVS_FOLDER) / experiment_path)


class Runs:
    def __init__(self, folder_path, done_hours, done_steps):
        csvs_path = Path(CSVS_FOLDER) / folder_path
        self.runs = [Run(experiment_path, done_hours=done_hours, done_steps=done_steps) for experiment_path in csvs_path.glob('*')]
        self.conf = defaultdict(set)
        self.folder = folder_path.name

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
        return new

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


class Run:
    def __init__(self, experiment_path, done_hours=None, done_steps=None):
        self.name = experiment_path.name
        self.conf = configuration_from_experiment_name(self.name)
        self.data_frame = pandas.read_csv(experiment_path, index_col=0)  # type: pandas.DataFrame

        self.done = None

        if done_hours is not None:
            self.done = self.data_frame.times.iat[-1] / 60 / 60 > done_hours

        if done_steps is not None:
            self.done = self.done or (self.data_frame.index[-1] >= done_steps)

    def metric(self, metric_name, last_n_steps=None, until_step=None, pooling='validation'):
        if until_step is not None:
            cutoff = self.data_frame[:until_step]
        else:
            cutoff = self.data_frame

        if last_n_steps is None:
            return cutoff[metric_name].iat[-1]
        else:
            cutoff = cutoff[last_n_steps:]

            if pooling == 'validation':
                max_rows = cutoff[cutoff[metric_name] == cutoff[metric_name].max()]
                max_validations = max_rows['rp' + metric_name]
                latest_max_validation = max_validations.iat[-1]
                return latest_max_validation
            elif pooling == 'max':
                return cutoff[metric_name].max()
            elif pooling == 'mean':
                return cutoff[metric_name].mean()

    def controller(self, last_n_steps=None, until_step=None, pooling='validation'):
        return self.metric('controller/mean', last_n_steps=last_n_steps, until_step=until_step, pooling=pooling)

    def __repr__(self):
        return "Run(done={}, {})".format(self.done, self.name)

    @property
    def id(self):
        return self.conf['id']

    def __call__(self, **kwargs):
        return all(self.conf[setting] == value for setting, value in kwargs.items())


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
    path = Path() / 'storage' / 'lisa3' / 'formal2'
    runs = Runs(path, done_hours=23, done_steps=100000)
    for run in runs:
        print(run)
test()