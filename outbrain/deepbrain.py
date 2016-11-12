import logging
import os

import coloredlogs
import keras
import luigi
import nose
import pandas
from sklearn import linear_model
import numpy as np
from sklearn.datasets.base import Bunch

from outbrain import task_utils, config
from outbrain.datasets import EventClicksDataset


class KerasClassifier(luigi.Task):
    test_run = luigi.parameter.BoolParameter(description='Run this as a test run, to evaluate the system')
    small = luigi.parameter.BoolParameter(description='Run as a small run (~1e8 rows) to check the code')
    vectorize = luigi.parameter.BoolParameter(description='Produce vectors, as an input for another system')

    def load(self):
        """Load in the vectors produced with the vectorize argument"""
        assert self.vectorize, 'Must be set up with the "vectorize" argument set true'
        assert self.complete(), 'Must run the KerasClassifier with "vectorize" first'
        f = open(self.output().path, 'rb')
        data = np.load(f)
        return data['train_vecs'], data['test_vecs']

    def requires(self):
        return EventClicksDataset(test_run=self.test_run, small=self.small)

    def output(self):
        if self.vectorize:
            return luigi.LocalTarget(config.working_path('keras-vectors-{}-{}.npz'.format(self.test_run, self.small)))

        if self.test_run:
            return luigi.s3.S3Target('s3://riri-machine-learning/outbrain-tests/{}.csv'.format(type(self).__name__))
        else:
            return luigi.s3.S3Target('s3://riri-machine-learning/outbrain-results/{}.csv'.format(type(self).__name__))

    def model(self):
        pass

    def run(self):
        coloredlogs.install(level=logging.INFO)
        logging.info('Gathering data')
        train_data, test_data = self.requires().load()

        max_documents = max(train_data.document_id.max(), test_data.document_id.max())
        max_ads = max(train_data.ad_id.max(), test_data.ad_id.max())
        max_users = max(train_data.uuid.max(), test_data.uuid.max())

        settings = Bunch(
            embedding_size=4
        )

        def build_embedding(largest_ix):
            k_inpt = keras.layers.Input([1])
            k_reshaped = keras.layers.Reshape([1, 1])(k_inpt)
            k_embedding = keras.layers.Embedding(
                largest_ix + 1, settings.embedding_size,
                dropout=0.5
            )(k_reshaped)
            k_flattened = keras.layers.Flatten()(k_embedding)
            return k_inpt, k_flattened

        # Document embedding
        k_doc_input, k_doc_embedding = build_embedding(max_documents)
        k_ad_input, k_ad_embedding = build_embedding(max_ads)
        k_user_input, k_user_embedding = build_embedding(max_users)

        merged_embeddings = keras.layers.merge([k_doc_embedding, k_ad_embedding, k_user_embedding], mode='concat')

        nn = keras.layers.Dropout(0.5)(keras.layers.Dense(32, activation='relu')(merged_embeddings))
        nn = keras.layers.Dense(16, activation='relu')(nn)
        nn = keras.layers.Dense(1, activation='sigmoid')(nn)

        Xs = [train_data.document_id.values, train_data.ad_id.values, train_data.uuid.values]
        test_Xs = [test_data.document_id.values, test_data.ad_id.values, test_data.uuid.values]
        ys = train_data.clicked.values[:, None]
        test_ys = test_data.clicked.values[:, None]
        sample_xs = [x[:20] for x in Xs]

        model = keras.models.Model([k_doc_input, k_ad_input, k_user_input], nn)
        # Check for a crash or size mismatch before the compile.
        nose.tools.assert_equals(model.predict(sample_xs).shape, (20, 1))
        model.compile('adam', 'binary_crossentropy')

        batch_size = 4096 * 64
        model.fit(Xs, ys, validation_data=(test_Xs, test_ys), nb_epoch=5, batch_size=batch_size)

        if self.vectorize:
            logging.info('Building the vectorizer model')
            vectorizer = keras.models.Model([k_doc_input, k_ad_input, k_user_input], merged_embeddings)
            logging.info('Vectorizing...')
            train_vecs = vectorizer.predict(Xs, verbose=1, batch_size=batch_size)
            test_vecs = vectorizer.predict(test_Xs, verbose=1, batch_size=batch_size)
            logging.info('Writing to output')
            self.output().makedirs()
            try:
                with open(self.output().path, 'wb') as f:
                    np.savez(f, train_vecs=train_vecs, test_vecs=test_vecs)
            except:
                os.remove(self.output().path)
                raise
            return

        logging.info('Predicting')
        predictions = pandas.Series(model.predict(test_Xs, verbose=1, batch_size=batch_size).flatten(),
                                    index=test_data.index).to_frame('prob')
        if self.test_run:
            predictions['clicked'] = test_data.clicked
            logging.info('Training discriminator')
            predictor = linear_model.LogisticRegression()
            predictor.fit(predictions[['prob']].as_matrix(), predictions[['clicked']])

            predictions['pred'] = predictor.predict(predictions[['prob']].as_matrix())
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id
            pa = task_utils.test_with_frame(predictions)
            ac = task_utils.test_accuracy_with_frame(predictions)
            logging.warning('Prediction accucracy: %f', pa)
            logging.warning('Accuracy Score: {}'.format(ac))

            with self.output().open('w') as f:
                f.write('Prediction accucracy: %f' % pa)
                f.write('\n')
                f.write('Accuracy Score: {}'.format(ac))
                f.write('\n')
        else:
            predictions['display_id'] = test_data.display_id
            predictions['ad_id'] = test_data.ad_id
            results = task_utils.retrieve_from_frame(predictions)
            with self.output().open('w') as f:
                task_utils.write_results(results, f)
