# Simple beat the benchmark.
# Basically, just calculate the probability that 
# a particular click will occur, based on the overall
# probability of a click.
# 

import luigi
import ml_metrics
import luigi.parameter
import pandas
from sklearn import cross_validation

from outbrain.data_sources import FetchS3ZipFile


class BeatTheBenchmark(luigi.Task):
    test_run = luigi.parameter.BoolParameter()

    def requires(self):
        return [FetchS3ZipFile(file_name='clicks_train.csv.zip'),
                FetchS3ZipFile(file_name='clicks_test.csv.zip')]

    def dataset(self):
        train_file, eval_file = self.requires()
        train_data = pandas.read_csv(train_file.output().path)
        if self.test_run:
            train_data, test_data = cross_validation.train_test_split(train_data)
        else:
            test_data = None
        eval_data = pandas.read_csv(eval_file.output().path)
        return train_data, test_data, eval_data

    def run(self):
        train_data, test_data, eval_data = self.dataset()

        click_count = train_data[train_data.clicked == 1].ad_id.value_counts()
        count_all = train_data.ad_id.value_counts()
        click_prob = (click_count / count_all).fillna(0)

        def srt(x):
            ad_ids = map(int, x.split())
            ad_ids = sorted(ad_ids, key=lambda k: click_prob.get(k, 0), reverse=True)
            return " ".join(map(str, ad_ids))

        working_data = test_data if self.test_run else eval_data
        subm = working_data.groupby('display_id').ad_id.apply(list)
        subm = subm.apply(srt)
        if self.test_run:
            print(subm.head())
            clicked = test_data[test_data.clicked == 1]
            correct_results = clicked.groupby('display_id').ad_id.apply(list)
            print(ml_metrics.average_precision.mapk(correct_results.values, subm, k=12))
        else:
            subm.to_csv("subm_1prob.csv", index=False)
