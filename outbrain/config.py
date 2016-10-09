import os

working_dir = '/mnt'


def working_path(*args):
    return os.path.join(working_dir, *args)
