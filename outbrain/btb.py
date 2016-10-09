# Simple beat the benchmark.
# Basically, just calculate the probability that 
# a particular click will occur, based on the overall
# probability of a click.
#
from collections import defaultdict

import luigi
import ml_metrics.average_precision
import luigi.parameter
import pandas
import pandas.io.sql
import logging
import numpy as np
import sqlite3
from tqdm import tqdm
import coloredlogs

from outbrain import task_utils
from outbrain.datasets import ClicksDataset


class BeatTheBenchmark(luigi.Task):
    test_run = luigi.parameter.BoolParameter()

    def requires(self):
        return ClicksDataset()

    def dataset(self):
        data_task = self.requires()
        if self.test_run:
            return data_task.load_train_clicks()
        else:
            return data_task.load_eval_clicks()

    def run(self):
        coloredlogs.install(level=logging.INFO)
        logging.info('Gathering datasets')
        train_data, test_data = self.dataset()

        logging.info('Computing stats')
        click_count = train_data[train_data.clicked == 1].ad_id.value_counts()
        count_all = train_data.ad_id.value_counts()
        click_prob = (click_count / count_all).fillna(0).to_frame('prob').reset_index()
        click_prob.columns = ['ad_id', 'prob']

        logging.info('Computed stats, building groups')

        merged_data = pandas.merge(test_data, click_prob, on='ad_id', how='left').fillna(0)

        if self.test_run:
            score = task_utils.test_with_frame(merged_data)
            logging.warning('Score: {}'.format(score))
        else:
            results = task_utils.retrieve_from_frame(merged_data)
            with open("subm_1prob.csv", 'w') as f:
                f.write('display_id,ad_id\n')
                for (display_id, ads) in results:
                    ads_s = ' '.join([str(ad) for ad in ads])
                    f.write('{},{}\n'.format(display_id, ads_s))
