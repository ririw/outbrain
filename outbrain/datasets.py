import logging
import os

import coloredlogs
import numpy as np
import joblib
import luigi
import luigi.parameter
import pandas
import pandas.io.sql
from sklearn import model_selection
from sklearn.datasets.base import Bunch

from outbrain import config, data_sources
from outbrain import task_utils


def read_csv(path):
    return pandas.read_csv(path)


def frame_work(frame, name):
    frame = frame.copy()
    logging.info('Writing file to ' + name)
    frame.to_pickle(config.working_path(name + '_clicks.pkl'))

    return


class ClicksDataset(luigi.Task):
    seed = luigi.parameter.IntParameter(default=12)

    def requires(self):
        return [
            data_sources.FetchS3ZipFile(file_name='clicks_train.csv.gz'),
            data_sources.FetchS3ZipFile(file_name='clicks_test.csv.gz')
        ]

    def output(self):
        return luigi.LocalTarget(config.working_path("ClicksDatasetSuccess"))

    def run(self):
        coloredlogs.install(level=logging.INFO)
        train_file, eval_file = self.requires()

        logging.info('Loading data')

        [all, eval] = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(read_csv)(train_file.output().path),
            joblib.delayed(read_csv)(eval_file.output().path),
        ])

        #train, test = model_selection.train_test_split(all, random_state=self.seed, test_size=0.10)
        split = all.shape[0] // 10 * 8
        train = all.iloc[:split]
        test = all.iloc[split:]
        train = train.copy()
        test = test.copy()

        joblib.Parallel(n_jobs=2)([
            joblib.delayed(frame_work)(all, 'all'),
            joblib.delayed(frame_work)(train, 'train'),
            joblib.delayed(frame_work)(test, 'test'),
            joblib.delayed(frame_work)(eval, 'eval')
        ])

        with self.output().open('w') as f:
            f.write('Done!')

    def load_train_clicks(self):
        assert self.complete()
        return (pandas.read_pickle(config.working_path("train_clicks.pkl")),
                pandas.read_pickle(config.working_path("test_clicks.pkl")))

    def load_train_groups(self):
        assert self.complete()
        return (pandas.read_pickle(config.working_path("train_groups.pkl")),
                pandas.read_pickle(config.working_path("test_groups.pkl")))

    def load_eval_clicks(self):
        assert self.complete()
        return (pandas.read_pickle(config.working_path("all_clicks.pkl")),
                pandas.read_pickle(config.working_path("eval_clicks.pkl")))

    def load_eval_groups(self):
        assert self.complete()
        return (pandas.read_pickle(config.working_path("all_groups.pkl")),
                pandas.read_pickle(config.working_path("eval_groups.pkl")))


class EventClicksDataset(luigi.Task):
    small = luigi.parameter.BoolParameter()
    test_run = luigi.parameter.BoolParameter()

    def requires(self):
        return [ClicksDataset(),
                data_sources.FetchS3ZipFile(file_name='events.csv.gz')]

    def directory(self):
        return config.working_path('event_clicks/{}/{}'.format(
            'small' if self.small else 'normal',
            'test' if self.test_run else 'full'
        ))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.directory(), 'done'))

    def run(self):
        coloredlogs.install(level=logging.INFO)
        logging.info('Gathering data')
        clicks_data, events_file = self.requires()
        logging.info('Gathering events')
        events = pandas.read_csv(events_file.output().path, dtype={
            'display_id': np.int64,
            'uuid': object,
            'document_id': np.int64,
            'timestamp': np.int64,
            'platform': object,
            'geo_location': object,
        })
        train_test_data = self.event_clicks(clicks_data, events, self.test_run, self.small)
        self.output().makedirs()
        train_test_data.train_data.to_pickle(os.path.join(self.directory(), 'train_data.pkl'))
        train_test_data.test_data.to_pickle(os.path.join(self.directory(), 'test_data.pkl'))

        with self.output().open('w') as f:
            pass

    def load(self):
        assert self.complete()
        test = pandas.read_pickle(os.path.join(self.directory(), 'test_data.pkl'))
        train = pandas.read_pickle(os.path.join(self.directory(), 'train_data.pkl'))
        return train, test

    @staticmethod
    def event_clicks(clicks_data, events, test_run, small=False):
        if test_run:
            train_clicks, test_clicks = clicks_data.load_train_clicks()
        else:
            train_clicks, test_clicks = clicks_data.load_eval_clicks()

        if small:
            train_clicks = train_clicks.head(1000000)
            test_clicks = test_clicks.head(1000000)

        logging.info('Building contexts')
        event_geo = task_utils.geo_expander(events.geo_location)
        events.timestamp += 1465876799998
        events.timestamp = events.timestamp.astype('datetime64[ms]')
        events['country'] = event_geo.country
        events['state'] = event_geo.state
        train_click_contexts = pandas.merge(train_clicks, events, on='display_id')
        test_click_contexts = pandas.merge(test_clicks, events, on='display_id')

        if 'clicked' not in test_click_contexts:
            test_click_contexts['clicked'] = 0
        train_click_contexts['country']
        train_data = train_click_contexts[['display_id', 'ad_id', 'document_id', 'platform',
                                           'uuid', 'country', 'state', 'clicked']]
        test_data = test_click_contexts[['display_id', 'ad_id', 'document_id', 'platform',
                                         'uuid', 'country', 'state', 'clicked']]
        del train_click_contexts, test_click_contexts, train_clicks, test_clicks, events

        field_cats = {}

        def convert_field_to_categories(field_name):
            train_data[field_name] = train_data[field_name].astype('category')
            field_cats[field_name] = train_data[field_name].cat.categories
            test_data[field_name] = test_data[field_name].astype('category', categories=field_cats[field_name])

            train_data[field_name] = (train_data[field_name].cat.codes + 1).astype(int)
            test_data[field_name] = (test_data[field_name].cat.codes + 1).astype(int)
            logging.warning('field %s min: %d max: %d',
                            field_name, train_data[field_name].min(), train_data[field_name].max())
            test_data.ix[test_data[field_name] == 1, field_name] = 0

        for col in train_data:
            if col == 'clicked' or col == 'display_id' or col == 'ad_id':
                continue
            convert_field_to_categories(col)

        return Bunch(
            test_data=test_data,
            train_data=train_data
        )
