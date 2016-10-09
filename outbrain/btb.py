# Simple beat the benchmark.
# Basically, just calculate the probability that 
# a particular click will occur, based on the overall
# probability of a click.
#
from collections import defaultdict

import luigi
import ml_metrics
import luigi.parameter
import pandas
import pandas.io.sql
import logging
import numpy as np
import sqlite3
from tqdm import tqdm
import coloredlogs

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

        def srt(ad_ids):
            # Shuffle the ad IDs
            ad_ids = ad_ids.values[np.random.permutation(len(ad_ids))]
            ad_ids = sorted(ad_ids, key=lambda k: click_prob.get(int(k), 0), reverse=True)
            return ad_ids

        def test_srt(frame):
            return list(frame.sort_values('clicked', ascending=False).ad_id.values)

        logging.info('Computed stats, building groups')

        merged_data = pandas.merge(test_data, click_prob, on='ad_id', how='left').fillna(0)
        con = sqlite3.connect(':memory:')
        logging.info('Writing to database')
        pandas.io.sql.to_sql(merged_data, 'ad', con=con)
        del merged_data
        con.execute('create index prob_ix on ad (prob)')
        con.execute('create index click_ix on ad (clicked)')

        logging.info('Querying...')
        results = defaultdict(list)
        cur = con.execute('select display_id, ad_id FROM ad ORDER BY prob DESC')
        for display_id, ad_id in tqdm(cur, total=test_data.shape[0]):
            results[display_id].append(ad_id)

        if self.test_run:
            true_results = defaultdict(list)
            cur = con.execute('select display_id, ad_id FROM ad ORDER BY clicked DESC')
            for display_id, ad_id in tqdm(cur, total=test_data.shape[0]):
                true_results[display_id].append(ad_id)
            keys = list(true_results.keys())
            pred = [true_results[k] for k in keys]
            results = [results[k] for k in keys]

            print(ml_metrics.average_precision.mapk(pred, results, k=12))
        else:
            with open("subm_1prob.csv", 'w') as f:
                f.write('display_id,ad_id\n')
                for (display_id, ads) in results:
                    ads_s = ' '.join([str(ad) for ad in ads])
                    f.write('{},{}\n'.format(display_id, ads_s))
