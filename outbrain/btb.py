# Simple beat the benchmark.
# Basically, just calculate the probability that 
# a particular click will occur, based on the overall
# probability of a click.
#

import logging

import coloredlogs
import luigi
import luigi.parameter
import luigi.s3
import pandas
import pandas.io.sql
from sklearn import linear_model
from tqdm import tqdm

from outbrain import task_utils
from outbrain.datasets import ClicksDataset
from outbrain.task_utils import write_results


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

    def output(self):
        if self.test_run:
            return []
        else:
            #return luigi.s3.S3Target('s3://riri-machine-learning/outbrain-results/beat-the-benchmark.csv')
            return luigi.LocalTarget('/mnt/{}.csv'.format(type(self).__name__))

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
        del test_data, click_prob, train_data, click_count

        if self.test_run:
            score = task_utils.test_with_frame(merged_data)
            logging.warning('Average Prediction Score: {}'.format(score))
            predictor = linear_model.LogisticRegression()
            # We cheat here and use the classifier on test data,
            # but I figure our bias in the choice of estimator is
            # so strong it won't matter.
            predictor.fit(merged_data[['prob']].as_matrix(), merged_data[['clicked']])
            merged_data['pred'] = predictor.predict(merged_data[['prob']].as_matrix())
            logging.warning('Accuracy Score: {}'.format(task_utils.test_accuracy_with_frame(merged_data)))
            logging.warning('Logloss Score: {}'.format(task_utils.test_logloss_with_frame(merged_data)))
        else:
            results = task_utils.retrieve_from_frame(merged_data)
            with self.output().open('w') as f:
                write_results(results, f)
