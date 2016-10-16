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

from outbrain import config
from outbrain.data_sources import FetchS3ZipFile
from outbrain.datasets import ClicksDataset
from outbrain.libffm_helpers import write_ffm_matrix, _file_version, write_vw_matrix

from outbrain.task_utils import test_accuracy_with_frame, retrieve_from_frame, write_results

assert _file_version == 9


class ExternalVWLikeClassifier(luigi.Task):
    test_run = luigi.parameter.BoolParameter()
    small = luigi.parameter.BoolParameter()

    def train(self, train_data, test_data):
        raise NotImplementedError

    def requires(self):
        return [ClicksDataset(), FetchS3ZipFile(file_name='events.csv.zip')]

    def run(self):
        coloredlogs.install(level=logging.INFO)
        logging.info('Gathering data')
        clicks_data, events_file = self.requires()
        if self.test_run:
            train_clicks, test_clicks = clicks_data.load_train_clicks()
            test_clicks = test_clicks
            train_clicks = train_clicks
        else:
            train_clicks, test_clicks = clicks_data.load_eval_clicks()

        if self.small:
            train_clicks = train_clicks.head(10000)
            test_clicks = test_clicks.head(10000)

        logging.info('Gathering events')
        events = pandas.read_csv(events_file.output().path, dtype={
            'display_id': np.int64,
            'uuid': object,
            'document_id': np.int64,
            'timestamp': np.int64,
            'platform': object,
            'geo_location': object,
        })

        logging.info('Building contexts')
        train_click_contexts = pandas.merge(train_clicks, events, on='display_id')
        test_click_contexts = pandas.merge(test_clicks, events, on='display_id')
        if 'clicked' not in test_click_contexts:
            test_click_contexts['clicked'] = 0
        train_data = train_click_contexts[['ad_id', 'document_id', 'platform', 'clicked']].copy()
        test_data = test_click_contexts[['ad_id', 'document_id', 'platform', 'clicked']].copy()
        del train_click_contexts, test_click_contexts, train_clicks, test_clicks, events

        field_cats = {}

        def convert_field_to_categories(field_name):
            train_data[field_name] = train_data[field_name].astype('category')
            field_cats[field_name] = train_data[field_name].cat.categories
            test_data[field_name] = test_data[field_name].astype('category', categories=field_cats[field_name])

            logging.warning('field %s min: %d max: %d',
                            field_name, field_cats[field_name].min(), field_cats[field_name].max())
            train_data[field_name] = (train_data[field_name].cat.codes + 1).astype(int)
            test_data[field_name] = (test_data[field_name].cat.codes + 1).astype(int)
            test_data.ix[test_data[field_name] == 1, field_name] = 0

        for col in train_data:
            if col == 'clicked':
                continue
            convert_field_to_categories(col)

        predictions = self.train(train_data, test_data)
        del train_data

        if self.test_run:
            predictions = predictions.to_frame('pred') > 0.5
            predictions['clicked'] = test_data.clicked
            logging.warning('Prediction accucracy: %f', test_accuracy_with_frame(predictions))
        else:
            predictions = predictions.to_frame('prob')
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id
            results = retrieve_from_frame(predictions)
            with open('./results.csv', 'w') as f:
                write_results(results, f)


class VWClassifier(ExternalVWLikeClassifier):
    @staticmethod
    def write_vw_rows(rows, target_file):
        clicked = rows.clicked.values
        document_id = rows.document_id.values
        ad_id = rows.ad_id.values
        platform = rows.platform.values

        write_vw_matrix(target_file, clicked, ad_id, document_id, platform)

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
        local['vw']['--hash', 'all', '--cache_file', cache_file, '--passes', 5, '-f', model_file, train_file] & FG
        logging.info('Classifying')
        local['vw']['-i', model_file, '-p', prediction_file, test_file] & FG
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
        write_ffm_matrix(target_file, clicked, ad_id, document_id, platform)

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
        local['./libffm/ffm-train']['-t', '50', '-s', str(multiprocessing.cpu_count()), train_file, model_file] & FG
        logging.info('Classifying')
        local['./libffm/ffm-predict'][test_file, model_file, prediction_file] & FG
        logging.warning('Wrote predictions to ' + prediction_file)
        logging.info('Reading predictions')
        predictions = pandas.read_csv(prediction_file, names=['prob'])
        return predictions.prob
