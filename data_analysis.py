import pandas as pd
import numpy as np
import tensorflow as tf
import os


def csvs(csv_path, must_include=None, must_exclude=None, joined=False, framed=True):
    model_names = [entry.name for entry in os.scandir(os.path.join(*csv_path))]

    if must_exclude is not None:
        model_names = [name for name in model_names if must_exclude not in name]

    if must_include is not None:
        model_names = [name for name in model_names if must_include in name]

    if framed:
        output = [pd.read_csv(os.path.join(*csv_path, model_name)) for model_name in model_names]
    elif joined:
        output = [os.path.join(*csv_path, model_name) for model_name in model_names]
    else:
        output = [csv_path + (model_name,) for model_name in model_names]

    return output


def bootstrap_interval(frames, n_samples, best_of_last_n):
    best_ones = pd.DataFrame([frame[-best_of_last_n:]['Value'].min() for frame in frames])
    # print(best_ones)
    samples = np.stack([best_ones.sample(10, replace=True).values.squeeze() for _ in range(n_samples)])
    # print(samples)
    means = pd.DataFrame(samples.mean(axis=1))
    # print(means)

    mean = best_ones.mean().values.squeeze()
    confidence = means.quantile([0.05, 0.95]).values.squeeze()
    print("mean: {} ({})".format(mean, confidence))
    return mean, confidence


def create_csvs(source_path, tag, must_include=None, must_exclude=None):
    model_names = [entry.name for entry in os.scandir(os.path.join(*source_path)) if entry.is_dir()]

    if must_exclude is not None:
        model_names = [name for name in model_names if must_exclude not in name]

    if must_include is not None:
        model_names = [name for name in model_names if must_include in name]

    output_path = ('csv',) + source_path[1:] + (tag.replace('/', '_'),)
    os.makedirs(os.path.join(*output_path))

    for model_name in model_names:
        event_file_name = [entry.name for entry in os.scandir(os.path.join(*source_path, model_name)) if entry.name.startswith('events')][0]

        output_file = output_path + ('{}.csv'.format(model_name),)

        wall_step_values = []

        session = tf.InteractiveSession()
        with session.as_default():
            for e in tf.train.summary_iterator(os.path.join(*source_path, model_name, event_file_name)):
                for v in e.summary.value:
                    if v.tag == tag:
                        wall_step_values.append((e.wall_time, e.step, v.simple_value))
        np.savetxt(os.path.join(*output_file), wall_step_values, delimiter=',', fmt='%-1f', header="Wall time,Step,Value", comments='')
        session.close()


def compare(frames1, frames2, best_of_last_n, at_steps):
    for step in at_steps:
        filtered1 = [frame[frame.Step <= step] for frame in frames1 if frame.Step.max() >= step]
        filtered2 = [frame[frame.Step <= step] for frame in frames2 if frame.Step.max() >= step]

        smallest_n = min(len(filtered1), len(filtered2))

        print("At step {} on {} samples:".format(step, smallest_n))
        bootstrap_interval(filtered1[:smallest_n], 10000, best_of_last_n)
        bootstrap_interval(filtered2[:smallest_n], 10000, best_of_last_n)
        print("----------------------------")


# csv_bootstrap_interval("run_hleak_v_{}-memoryless_True-id_{}-tag-controller_mean.csv", [(n + 1, n) for n in range(10)], 10000)
# csv_bootstrap_interval(csvs(('csv', 'lisa2', 'varia_hleak', 'controller_mean')), 10000)
# create_csvs(('storage', 'lisa2', 'varia_hleak'), 'controller/mean', must_include='True')
# create_csvs(('storage', 'lisa2', 'varia_bootstrap'), 'controller/mean')
#
compare(
    csvs(('csv', 'lisa2', 'varia_hleak', 'controller_mean'), must_include='True'),
    csvs(('csv', 'lisa2', 'varia_bootstrap', 'controller_mean')),
    best_of_last_n=40,
    at_steps=[50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
)
