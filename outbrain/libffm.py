"""
Things that use libffm.


1. LibFFMPrediction - predict using libffm, in the straightforward way, using:
  - document_id
  - ad_id
  - platform
  - uuid (ie, person. This may overfit)
"""
import logging
import os
import tempfile

import coloredlogs
import luigi
import multiprocessing
import numpy as np
import pandas
import shutil
from plumbum import local, FG
from sklearn import linear_model

from outbrain import config, task_utils, data_sources, datasets, libffm_helpers

assert libffm_helpers._file_version == 15


class ExternalVWLikeClassifier(luigi.Task):
    test_run = luigi.parameter.BoolParameter()
    small = luigi.parameter.BoolParameter()

    def train(self, train_data, test_data):
        raise NotImplementedError

    def requires(self):
        return datasets.EventClicksDataset(
            small=self.small,
            test_run=self.test_run
        )

    def output(self):
            if self.test_run:
                return []
            else:
                return luigi.s3.S3Target('s3://riri-machine-learning/outbrain-results/{}.csv'.format(type(self).__name__))

    def run(self):
        coloredlogs.install(level=logging.INFO)
        train_data, test_data = self.requires().load()

        predictions = self.train(train_data, test_data)
        del train_data

        if self.test_run:
            predictions = predictions.to_frame('prob')
            predictions['clicked'] = test_data.clicked
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id

            # We cheat here and use the classifier on test data,
            # but I figure our bias in the choice of estimator is
            # so strong it won't matter.
            predictor = linear_model.LogisticRegression()
            predictor.fit(predictions[['prob']].as_matrix(), predictions['clicked'])
            predictions['pred'] = predictor.predict(predictions[['prob']].as_matrix())
            logging.warning('Prediction accucracy: %f', task_utils.test_with_frame(predictions))
            logging.warning('Accuracy Score: {}'.format(task_utils.test_accuracy_with_frame(predictions)))
            logging.warning('Logloss Score: {}'.format(task_utils.test_logloss_with_frame(predictions)))

        else:
            predictions = predictions.to_frame('prob')
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id
            results = task_utils.retrieve_from_frame(predictions)
            with self.output().open('w') as f:
                task_utils.write_results(results, f)


class VWClassifier(ExternalVWLikeClassifier):
    @staticmethod
    def write_vw_rows(rows, target_file):
        clicked = rows.clicked.values
        document_id = rows.document_id.values
        ad_id = rows.ad_id.values
        platform = rows.platform.values
        user_id = rows.uuid.values

        libffm_helpers.write_vw_matrix(target_file, clicked, ad_id, document_id, platform, user_id)

    def train(self, train_data, test_data):
        logging.info('Preparing files')
        shutil.rmtree(config.working_path('vw_working'), ignore_errors=True)
        os.makedirs(config.working_path('vw_working'), exist_ok=True)
        train_file = tempfile.mktemp(dir=config.working_path('vw_working'), prefix='train_data.')
        model_file = tempfile.mktemp(dir=config.working_path('vw_working'), prefix='model.')
        test_file = tempfile.mktemp(dir=config.working_path('vw_working'), prefix='test_data.')
        prediction_file = tempfile.mktemp(dir=config.working_path('vw_working'), prefix='prediction.')
        cache_file = tempfile.mktemp(dir=config.working_path('vw_working'), prefix='cache.')

        logging.info('Writing to the files')
        self.write_vw_rows(train_data, train_file)
        self.write_vw_rows(test_data, test_file)

        logging.info('Training')
        local['vw'][
            '-k --cache_file {} -b 24 '
            '-q ua -q ud -q da --rank 8 '
            '--l2 0.005 -l 0.01 '
            '--passes 100 '
            '--decay_learning_rate 0.97 --power_t 0 '
            '--loss_function logistic --link logistic -f {} {}'.format(cache_file, model_file, train_file).split(' ')
        ] & FG
        logging.info('Classifying')
        local['vw']['-i', model_file,
                    '-p', prediction_file,
                    '-t', # Test mode
                    test_file] & FG
        logging.warning('Wrote predictions to ' + prediction_file)
        logging.info('Reading predictions')
        predictions = pandas.read_csv(prediction_file, names=['prob'])
        return predictions.prob


class LibFFMClassifier(ExternalVWLikeClassifier):
    @staticmethod
    def write_libffm_rows(rows, target_file):
        clicked = rows.clicked.values
        document_id = rows.document_id.values
        ad_id = rows.ad_id.values
        platform = rows.platform.values.astype(int)
        libffm_helpers.write_ffm_matrix(target_file, clicked, ad_id, document_id, platform)

    def train(self, train_data, test_data):
        logging.info('Preparing files')
        shutil.rmtree(config.working_path('libffm_working'), ignore_errors=True)
        os.makedirs(config.working_path('libffm_working'), exist_ok=True)
        train_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='train_data.')
        model_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='model.')
        test_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='test_data.')
        prediction_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='prediction.')

        logging.info('Writing to the files')
        self.write_libffm_rows(train_data, train_file)
        self.write_libffm_rows(test_data, test_file)

        with local.cwd("libffm"):
            local['make']('-B')
        logging.info('Training')
        local['./libffm/ffm-train'][
            '-t', '100',  # Number of passes
            '-k', '16',  # Latent factors
            '-r', '0.5', # Learning rate
            '-s', str(multiprocessing.cpu_count()),  # Number of threads
            train_file, model_file] & FG
        logging.info('Classifying')
        local['./libffm/ffm-predict'][test_file, model_file, prediction_file] & FG
        logging.warning('Wrote predictions to ' + prediction_file)
        logging.info('Reading predictions')
        predictions = pandas.read_csv(prediction_file, names=['prob'])
        return predictions.prob
