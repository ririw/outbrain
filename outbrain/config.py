import os

working_dir = '/mnt'


lightgbm = '../LightGBM/lightgbm'


def working_path(*args):
    return os.path.join(working_dir, *args)
