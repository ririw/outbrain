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
import numpy as np
import pandas
import shutil
from plumbum import local, FG
from tqdm import tqdm

from outbrain import config
from outbrain.data_sources import FetchS3ZipFile
from outbrain.datasets import ClicksDataset
from outbrain.libffm_helpers import write_ffm_matrix, _file_version

from outbrain.task_utils import test_accuracy_with_frame, retrieve_from_frame, write_results

assert _file_version == 4


def write_libffm_row(rows, target_file):
    clicked = rows.clicked.values
    document_id = rows.document_id.values
    ad_id = rows.ad_id.values
    uuid = rows.uuid.values.astype(int)
    platform = rows.platform.values.astype(int)
    write_ffm_matrix(target_file, clicked, ad_id, document_id, platform, uuid)


class LibFFMRun(luigi.Task):
    test_run = luigi.parameter.BoolParameter()

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
        train_data = train_click_contexts[['ad_id', 'document_id', 'uuid', 'platform', 'clicked']].copy()
        test_data = test_click_contexts[['ad_id', 'document_id', 'uuid', 'platform', 'clicked']].copy()

        train_data.platform = train_data.platform.astype('category')
        train_data.uuid = train_data.uuid.astype('category')
        platform_cats = train_data.platform.cat.categories
        uuid_cats = train_data.uuid.cat.categories
        test_data.platform = test_data.platform.astype('category', categories=platform_cats)
        test_data.uuid = test_data.uuid.astype('category', categories=uuid_cats)

        train_data.platform = train_data.platform.cat.codes
        train_data.uuid = train_data.uuid.cat.codes
        test_data.platform = test_data.platform.cat.codes
        test_data.uuid = test_data.uuid.cat.codes
        test_data.ix[test_data.platform == -1, 'platform'] = 0
        test_data.ix[test_data.uuid == -1, 'uuid'] = 0

        logging.info('Preparing files')
        shutil.rmtree(config.working_path('libffm_working'), ignore_errors=True)
        os.makedirs(config.working_path('libffm_working'), exist_ok=True)
        train_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='train_data.')
        test_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='test_data.')
        model_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='model.')
        prediction_file = tempfile.mktemp(dir=config.working_path('libffm_working'), prefix='prediction.')
        logging.info('Writing to files')
        write_libffm_row(train_data, train_file)
        write_libffm_row(test_data, test_file)
        with local.cwd("libffm"):
            local['make']('-B')

        logging.info('Training')
        local['./libffm/ffm-train'][train_file, model_file] & FG
        logging.info('Classifying')
        local['./libffm/ffm-predict'][test_file, model_file, prediction_file] & FG
        logging.warning('Wrote predictions to ' + prediction_file)

        if self.test_run:
            predictions = pandas.read_csv(prediction_file).to_frame('pred') > 0.5
            predictions['clicked'] = test_data.clicked
            logging.warning('Prediction accucracy: ', test_accuracy_with_frame(predictions))
        else:
            predictions = pandas.read_csv(prediction_file).to_frame('prob')
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id
            results = retrieve_from_frame(predictions)
            with open('./results.csv', 'w') as f:
                write_results(results, f)
