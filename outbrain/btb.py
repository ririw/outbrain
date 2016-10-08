# Simple beat the benchmark.
# Basically, just calculate the probability that 
# a particular click will occur, based on the overall
# probability of a click.
# 

import luigi
import luigi.parameter
import pandas

from outbrain.data_sources import FetchS3ZipFile

class BeatTheBenchmark(luigi.Task):
    test_run = luigi.parameter.BoolParameter()

    def requires(self):
        return [FetchS3ZipFile(file_name='clicks_train.csv.zip'),
                FetchS3ZipFile(file_name='sample_submission.csv.zip')]

    def run(self):
        train_file, test_file = self.requires()

        train_data = pandas.read_csv(train_file.output().path)
        click_count = train_data[train_data.clicked == 1].ad_id.value_counts()
        count_all = train_data.ad_id.value_counts()
        click_prob = (click_count / count_all).fillna(0)

        def srt(x):
                ad_ids = map(int, x.split())
                ad_ids = sorted(ad_ids, key=lambda k: click_prob.get(k, 0), reverse=True)
                return " ".join(map(str,ad_ids))

        subm = pandas.read_csv(test_file.output().path)
        subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
        subm.to_csv("subm_1prob.csv", index=False)
