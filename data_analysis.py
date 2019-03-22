import pandas as pd
import numpy as np


def csv_bootstrap_interval(file_name, arguments, n_samples):
    strings = ['csv/' + file_name.format(*argument) for argument in arguments]
    best_ones = pd.DataFrame([pd.read_csv(str)[-40:]['Value'].min() for str in strings])
    print(best_ones)
    samples = np.stack([best_ones.sample(10, replace=True).values.squeeze() for _ in range(n_samples)])
    # print(samples)
    means = pd.DataFrame(samples.mean(axis=1))
    # print(means)
    print("mean: {}".format(best_ones.mean().values.squeeze()))
    print("confidence: {}".format(means.quantile([0.05, 0.95]).values.squeeze()))


csv_bootstrap_interval("run_hleak_v_{}-memoryless_True-id_{}-tag-controller_mean.csv", [(n + 1, n) for n in range(10)], 10000)
