# Simple beat the benchmark.
# Basically, just calculate the probability that 
# a particular click will occur, based on the overall
# probability of a click.
#
from collections import defaultdict

import luigi
import ml_metrics
import luigi.parameter
import joblib
import pandas
import logging
import numpy as np
from tqdm import tqdm
import coloredlogs

from outbrain.data_sources import FetchS3ZipFile
from outbrain.datasets import ClicksDataset


class BeatTheBenchmark(luigi.Task):
    test_run = luigi.parameter.BoolParameter()

    def requires(self):
        return ClicksDataset()

    def dataset(self):
        data_task = self.requires()
        if self.test_run:
            return data_task.load_train_clicks()[0], data_task.load_train_groups()[1]
        else:
            return data_task.load_eval_clicks()[0], data_task.load_eval_groups()[1]

    def run(self):
        coloredlogs.install(level=logging.INFO)
        logging.info('Gathering datasets')
        train_data, test_data = self.dataset()

        logging.info('Computing stats')
        click_count = train_data[train_data.clicked == 1].ad_id.value_counts()
        count_all = train_data.ad_id.value_counts()
        click_prob = (click_count / count_all).fillna(0)

        def srt(ad_ids):
            # Shuffle the ad IDs
            ad_ids = ad_ids[np.random.permutation(len(ad_ids))]
            ad_ids = sorted(ad_ids, key=lambda k: click_prob.get(k, 0), reverse=True)
            return ad_ids

        logging.info('Computed stats, building groups')

        results = []
        for l in tqdm(test_data, total=test_data.shape[0]):
            results.append(srt(l))
        results = pandas.Series(results, index=test_data.index)

        if self.test_run:
            print(ml_metrics.average_precision.mapk(test_data.values, results, k=12))
        else:
            results.to_csv("subm_1prob.csv", index=False)
