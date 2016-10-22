import logging

import coloredlogs
import keras
import nose
import numpy as np
import luigi
import pandas
from sklearn import linear_model
from sklearn.datasets.base import Bunch

from outbrain import datasets
from outbrain import task_utils
from outbrain.data_sources import FetchS3ZipFile
from outbrain.datasets import ClicksDataset


class KerasClassifier(luigi.Task):
    test_run = luigi.parameter.BoolParameter()
    small = luigi.parameter.BoolParameter()

    def requires(self):
        return [ClicksDataset(), FetchS3ZipFile(file_name='events.csv.zip')]

    def output(self):
            if self.test_run:
                return []
            else:
                return luigi.s3.S3Target('s3://riri-machine-learning/outbrain-results/{}.csv'.format(type(self).__name__))

    def model(self):
        pass

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

        train_test_data = datasets.DatasetViews.event_clicks(clicks_data, events, self.test_run, self.small)
        train_data = train_test_data.train_data
        test_data = train_test_data.test_data

        max_documents = train_data.document_id.max()
        max_users = train_data.uuid.max()
        max_ads = train_data.ad_id.max()

        settings = Bunch(
            embedding_size=4
        )

        def build_embedding(largest_ix):
            k_inpt = keras.layers.Input([1])
            k_reshaped = keras.layers.Reshape([1, 1])(k_inpt)
            k_embedding = keras.layers.Embedding(
                largest_ix + 1, settings.embedding_size, dropout=0.5
            )(k_reshaped)
            k_flattened = keras.layers.Flatten()(k_embedding)
            return k_inpt, k_flattened

        # Document embedding

        k_doc_input, k_doc_embedding = build_embedding(max_documents)
        k_users_input, k_users_embedding = build_embedding(max_users)
        k_ad_input, k_ad_embedding = build_embedding(max_ads)

        merged_embeddings = keras.layers.merge([k_doc_embedding, k_users_embedding, k_ad_embedding],
                                               mode='concat')

        nn = keras.layers.Dense(512, activation='relu')(merged_embeddings)
        nn = keras.layers.Dropout(0.5)(nn)

        nn = keras.layers.Dense(256, activation='relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)

        nn = keras.layers.Dense(128, activation='relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)

        nn = keras.layers.Dense(64, activation='relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)

        nn = keras.layers.Dense(1, activation='sigmoid')(nn)

        Xs = [train_data.document_id.values, train_data.uuid.values, train_data.ad_id.values]
        test_Xs = [test_data.document_id.values, test_data.uuid.values, test_data.ad_id.values]
        ys = train_data.clicked.values[:, None]
        test_ys = test_data.clicked.values[:, None]
        sample_xs = [x[:20] for x in Xs]

        model = keras.models.Model([k_doc_input, k_users_input, k_ad_input], nn)
        # Check for a crash or size mismatch before the compile.
        nose.tools.assert_equals(model.predict(sample_xs).shape, (20, 1))
        model.compile('adam', 'binary_crossentropy')

        model.fit(Xs, ys, validation_data=(test_Xs, test_ys), nb_epoch=20, batch_size=512)

        if self.test_run:
            predictions = pandas.Series(model.predict(test_Xs).flatten(), index=test_data.index).to_frame('prob')
            predictions['clicked'] = test_data.clicked
            predictor = linear_model.LogisticRegression()
            predictor.fit(predictions[['prob']].as_matrix(), predictions[['clicked']])
            predictions['pred'] = predictor.predict(predictions[['prob']].as_matrix())
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id
            print(predictions.dtypes)
            logging.warning('Prediction accucracy: %f', task_utils.test_with_frame(predictions))
            logging.warning('Accuracy Score: {}'.format(task_utils.test_accuracy_with_frame(predictions)))
        else:
            predictions = pandas.Series(model.predict(test_Xs).flatten(), index=test_data.index).to_frame('prob')
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id
            results = task_utils.retrieve_from_frame(predictions)
            with self.output().open('w') as f:
                task_utils.write_results(results, f)
