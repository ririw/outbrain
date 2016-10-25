import luigi
import luigi.file
import luigi.s3
from plumbum import local
import outbrain.config
import os


class FetchS3Data(luigi.Task):
    def requires(self):
        yield FetchS3ZipFile(file_name='clicks_test.csv.gz')
        yield FetchS3ZipFile(file_name='clicks_train.csv.gz')
        yield FetchS3ZipFile(file_name='documents_categories.csv.gz')
        yield FetchS3ZipFile(file_name='documents_entities.csv.gz')
        yield FetchS3ZipFile(file_name='documents_meta.csv.gz')
        yield FetchS3ZipFile(file_name='documents_topics.csv.gz')
        yield FetchS3ZipFile(file_name='events.csv.gz')
        yield FetchS3ZipFile(file_name='page_views_sample.csv.gz')
        yield FetchS3ZipFile(file_name='promoted_content.csv.gz')
        yield FetchS3ZipFile(file_name='sample_submission.csv.gz')

    def run(self):
        pass


class FetchPageViews(luigi.Task):
    def requires(self):
        yield FetchS3ZipFile(file_name='page_views.csv.gz')


class FetchS3ZipFile(luigi.Task):
    file_name = luigi.Parameter()

    def output(self):
        outpath = outbrain.config.working_path(self.file_name)
        return luigi.LocalTarget(outpath)

    def run(self):
        client = luigi.s3.S3Client()
        path = self.output().path
        client.get('s3://riri-machine-learning/outbrain/{}'.format(self.file_name), path)
