"""
A classifier built around the recently released LightGBM.

TODO:

 - Investigate its ranking stuff
 - Play with configs
 - Use the vectorization in the deepbrain file
 - Basically, merge together all data-sources.
"""
import logging
import os
import tempfile

import coloredlogs
import luigi
import multiprocessing
import pandas
import shutil
from plumbum import local, FG
from sklearn import linear_model

from outbrain import config
from outbrain import task_utils
from outbrain.datasets import EventClicksDataset
from outbrain.deepbrain import KerasClassifier
import outbrain.io_helpers
import outbrain.config

assert outbrain.io_helpers._file_version == 16


class LightGBTClassifier(luigi.Task):
    test_run = luigi.parameter.BoolParameter(description='Run this as a test run, to evaluate the system')
    small = luigi.parameter.BoolParameter(description='Run as a small run (~1e8 rows) to check the code')

    def requires(self):
        return [EventClicksDataset(test_run=self.test_run, small=self.small),
                KerasClassifier(test_run=self.test_run, small=self.small, vectorize=True)]

    def output(self):
        return luigi.s3.S3Target('s3://riri-machine-learning/outbrain-results/{}.csv'.format(type(self).__name__))

    def run(self):
        coloredlogs.install(level=logging.INFO)
        event_clicks, keras_vecs = self.requires()
        logging.info('loading clicks')
        train_clicks, test_clicks = event_clicks.load()
        logging.info('loading vecs')
        train_vecs, test_vecs = keras_vecs.load()

        # Make frames of all vecs, then pull out clicked, combine all
        # the vectors together, and put clicked as the leading column.
        # it needs to lead to match the format LightGBM expects.
        logging.info('prepping data')
        train_vecs = pandas.DataFrame(train_vecs, index=train_clicks.index)
        test_vecs = pandas.DataFrame(test_vecs, index=test_clicks.index)
        train_Y = train_clicks.clicked
        test_Y = test_clicks.clicked
        train_X = pandas.concat([train_clicks, train_vecs], 1).drop('clicked', 1)
        test_X = pandas.concat([test_clicks, test_vecs], 1).drop('clicked', 1)
        del train_vecs, test_vecs

        train_X.insert(0, 'clicked', train_Y)
        test_X.insert(0, 'clicked', test_Y)

        os.makedirs(config.working_path("lightgbm_working"), exist_ok=True)
        shutil.rmtree(config.working_path("lightgbm_working"))
        os.makedirs(config.working_path("lightgbm_working"), exist_ok=True)
        train_file = tempfile.mktemp(dir=config.working_path('lightgbm_working'), prefix='train_data.')
        model_file = tempfile.mktemp(dir=config.working_path('lightgbm_working'), prefix='model.')
        config_file = tempfile.mktemp(dir=config.working_path('lightgbm_working'), prefix='config.')
        test_file = tempfile.mktemp(dir=config.working_path('lightgbm_working'), prefix='test_data.')
        prediction_file = tempfile.mktemp(dir=config.working_path('lightgbm_working'), prefix='prediction.')

        logging.info('Writing out training data')
        train_X.to_csv(train_file, header=False, index=False)
        logging.info('Writing out testing data')
        test_X.to_csv(test_file, header=False, index=False)
        del train_X, test_X, train_Y, test_Y

        with open(config_file, 'w') as f:
            f.write(lightgbm_train_config.format(
                training_data=train_file,
                show_test_data='' if not self.test_run else 'valid_data=' + test_file,
                num_cores=multiprocessing.cpu_count(),
                model_file=model_file,
            ))

        logging.info('Training...')
        local[config.lightgbm]['config=' + config_file] & FG
        logging.info('Predicting...')
        local[config.lightgbm]['task=predict', 'data=' + test_file,
                               'input_model=' + model_file, 'output_result=' + prediction_file] & FG

        logging.info('Reading predictions')
        predictions = pandas.read_csv(prediction_file, names=['prob'])
        if self.test_run:
            predictions['clicked'] = test_clicks.clicked
            predictions['display_id'] = test_clicks.display_id
            predictions['ad_id'] = test_clicks.ad_id

            predictor = linear_model.LogisticRegression()
            predictor.fit(predictions[['prob']].as_matrix(), predictions['clicked'])
            predictions['pred'] = predictor.predict(predictions[['prob']].as_matrix())
            logging.warning('Prediction accucracy: %f', task_utils.test_with_frame(predictions))
            logging.warning('Accuracy Score: {}'.format(task_utils.test_accuracy_with_frame(predictions)))
            logging.warning('Logloss Score: {}'.format(task_utils.test_logloss_with_frame(predictions)))
        else:
            predictions['display_id'] = test_clicks.display_id
            predictions['ad_id'] = test_clicks.ad_id
            results = task_utils.retrieve_from_frame(predictions)
            with self.output().open('w') as f:
                task_utils.write_results(results, f)



lightgbm_train_config = """
task = train
boosting_type = dart
# Could also be lambdarank
objective = binary
# ndcg , default metric for lambdarank
# binary_logloss , default metric for binary
metric = binary_logloss
#metric = ndcg
# evaluation position for ndcg metric, alias : ndcg_at
ndcg_eval_at = 1,3,5
metric_freq = 1
is_training_metric = true
max_bin = 255
data = {training_data}
{show_test_data}
num_trees = 100
learning_rate = 0.05
num_leaves = 511
tree_learner = serial
num_threads = {num_cores}
feature_fraction = 0.8
bagging_freq = 1
bagging_fraction = 0.8
min_data_in_leaf = 1000
min_sum_hessian_in_leaf = 5.0
#is_enable_sparse = true
use_two_round_loading = false
is_save_binary_file = false
output_model = {model_file}
num_machines = 1
local_listen_port = 12400
"""

