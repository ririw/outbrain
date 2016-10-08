import luigi
from plumbum import local
import luigi.file
import config
import os

class FetchS3ZipFile(luigi.Task):
    file_name = luigi.Parameter()

    def output(self):
        assert self.file_name.endswith('.zip')
        f = self.file_name[:-4]
        outpath = os.path.join(config.local, f)
        return luigi.LocalTarget(outpath)

    def run(self):
        client = luigi.s3.S3Client()
        path = self.output().name
        out = client.get('s3://riri-machine-learning/{}'.format(self.file_name), path)
        with local.cwd(config.working_dir):
            local['unzip'](self.file_name)
