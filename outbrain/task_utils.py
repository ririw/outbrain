import logging
from collections import defaultdict

import ml_metrics.average_precision
import pandas.io.sql
from sklearn import metrics
import sqlite3

from tqdm import tqdm


def test_with_frame(merged_data):
    con = sqlite3.connect(':memory:')
    logging.info('Writing to database')
    pandas.io.sql.to_sql(merged_data, 'ad', con=con)
    con.execute('create index prob_ix on ad (prob)')
    con.execute('create index click_ix on ad (clicked)')

    logging.info('Querying...')
    results = defaultdict(list)
    cur = con.execute('select display_id, ad_id FROM ad ORDER BY prob DESC')
    for display_id, ad_id in tqdm(cur, desc='reading predictions', total=merged_data.shape[0]):
        results[display_id].append(ad_id)

    cur = con.execute('select display_id, ad_id FROM ad WHERE clicked = 1')
    true_results = defaultdict(list)
    for display_id, ad_id in tqdm(cur, desc='reading true values', total=merged_data.shape[0]):
        true_results[display_id].append(ad_id)
    con.close()
    pred = [true_results[k] for k in true_results.keys()]
    results = [results[k] for k in true_results.keys()]

    return ml_metrics.average_precision.mapk(pred, results, k=2)


def test_accuracy_with_frame(merged_database):
    return metrics.accuracy_score(merged_database.clicked, merged_database.pred)


def retrieve_from_frame(merged_data):
    con = sqlite3.connect(':memory:')
    logging.info('Writing to database')
    pandas.io.sql.to_sql(merged_data, 'ad', con=con)
    con.execute('create index prob_ix on ad (prob)')

    logging.info('Querying...')
    results = defaultdict(list)
    cur = con.execute('select display_id, ad_id FROM ad ORDER BY prob DESC')
    for display_id, ad_id in tqdm(cur, desc='reading predictions', total=merged_data.shape[0]):
        results[display_id].append(ad_id)

    con.close()
    return results


def write_results(results, handle):
    handle.write('display_id,ad_id\n')
    for (display_id, ads) in tqdm(results.items(), total=len(results), desc='writing results'):
        ads_s = ' '.join([str(ad) for ad in ads])
        handle.write('{},{}\n'.format(display_id, ads_s))
