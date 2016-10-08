import luigi
import luigi.file
import luigi.s3
from plumbum import local
import outbrain.config
import os

class FetchS3Data(luigi.Task):
    def requires(self):
        yield FetchS3ZipFile(name='clicks_test.csv.zip')
        yield FetchS3ZipFile(name='clicks_train.csv.zip')
        yield FetchS3ZipFile(name='documents_categories.csv.zip')
        yield FetchS3ZipFile(name='documents_entities.csv.zip')
        yield FetchS3ZipFile(name='documents_meta.csv.zip')
        yield FetchS3ZipFile(name='documents_topics.csv.zip')
        yield FetchS3ZipFile(name='events.csv.zip')
        yield FetchS3ZipFile(name='page_views.csv.zip')
        yield FetchS3ZipFile(name='page_views_sample.csv.zip')
        yield FetchS3ZipFile(name='promoted_content.csv.zip')
        yield FetchS3ZipFile(name='sample_submission.csv.zip')

    def run(self):
        pass


class FetchS3ZipFile(luigi.Task):
    file_name = luigi.Parameter()

    def output(self):
        assert self.file_name.endswith('.zip')
        f = self.file_name[:-4]
        outpath = os.path.join(outbrain.config.local, f)
        return luigi.LocalTarget(outpath)

    def run(self):
        client = luigi.s3.S3Client()
        path = self.output().path
        out = client.get('s3://riri-machine-learning/{}'.format(self.file_name), path)
        with local.cwd(outbrain.config.working_dir):
            local['unzip'](self.file_name)


