import logging
from collections import defaultdict

from outbrain import config, data_sources
import luigi
import luigi.parameter
import joblib
import sqlite3
import pandas
import pandas.io.sql
from sklearn import model_selection
import coloredlogs
from tqdm import tqdm


def read_csv(path):
    return pandas.read_csv(path)


def frame_work(frame, name):
    frame = frame.copy()
    logging.info('Writing file to ' + name)
    frame.to_pickle(config.working_path(name + '_clicks.pkl'))

    return
    con = sqlite3.connect(':memory')
    pandas.io.sql.to_sql(frame, 'ad', con=con)
    con.execute('create index on ad using (display_id, clicked)')

    logging.info('grouping ' + name)
    if 'clicked' in frame:
        result = defaultdict(list)
        rows = con.execute('''
            SELECT display_id,
                   ad_id
              FROM ad
          ORDER BY display_id, clicked ASC
        ''')
        for display_id, ad_id in rows:
            result[display_id].append(ad_id)
        result = pandas.Series(result)
    else:
        result = defaultdict(list)
        rows = con.execute('''
            SELECT display_id,
                   ad_id
              FROM ad
          ORDER BY display_id
        ''')
        for display_id, ad_id in rows:
            result[display_id].append(ad_id)
        result = pandas.Series(result)
    result.to_pickle(config.working_path(name + '_groups.pkl'))


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

        train, test = model_selection.train_test_split(all, random_state=self.seed, test_size=0.5)
        train = train.copy()
        test = test.copy()

        joblib.Parallel(n_jobs=2)([
            joblib.delayed(frame_work)(all,   'all'),
            joblib.delayed(frame_work)(train, 'train'),
            joblib.delayed(frame_work)(test,  'test'),
            joblib.delayed(frame_work)(eval,  'eval')
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
