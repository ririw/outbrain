import logging
from collections import defaultdict

from outbrain import config, data_sources
import luigi
import luigi.parameter
import joblib
import pandas
from sklearn import model_selection
import coloredlogs
from tqdm import tqdm


def read_csv(path):
    return pandas.read_csv(path)


def frame_work(frame, name):
    frame = frame.copy()
    def sort_and_list(f):
        return list(f.sort_values('clicked', ascending=False).ad_id)

    logging.info('Writing file to ' + name)
    frame.to_pickle(config.working_path(name + '_clicks.pkl'))
    logging.info('grouping ' + name)
    frame.groupby('display_id').apply(sort_and_list)
    frame.to_pickle(config.working_path(name + '_groups.pkl'))


class ClicksDataset(luigi.Task):
    seed = luigi.parameter.IntParameter(default=12)

    def requires(self):
        return [
            data_sources.FetchS3ZipFile(file_name='clicks_train.csv.zip'),
            data_sources.FetchS3ZipFile(file_name='clicks_test.csv.zip')
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

        train, test = model_selection.train_test_split(all, random_state=self.seed)
        train = train.copy()
        test = test.copy()

        joblib.Parallel(n_jobs=-1)([
            joblib.delayed(frame_work)(all,   'all'),
            joblib.delayed(frame_work)(train, 'train'),
            joblib.delayed(frame_work)(test,  'test'),
            joblib.delayed(frame_work)(eval,  'eval')
        ])

        with self.output().open() as f:
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
