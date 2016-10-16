cimport libc.stdio
cimport libc.stdlib
import numpy as np
cimport numpy as np


# Quickly write to a LIBFFM compatible file.
# This saves *masses* of time by looping in C
# and using raw C io.
def write_ffm_matrix(str target_file,
                 np.ndarray[np.int_t, ndim=1] clicked,
                 np.ndarray[np.int_t, ndim=1] ad_id,
                 np.ndarray[np.int_t, ndim=1] document_id,
                 np.ndarray[np.int_t, ndim=1] platform):
    f = libc.stdio.fopen(target_file.encode('ascii'), 'w')
    cdef int c, a, d, p, i

    for i in range(0, clicked.shape[0]):
        c = clicked[i]
        a = ad_id[i]
        d = document_id[i]
        p = platform[i]
        libc.stdio.fprintf(f, "%d 0:a%d:1 1:b%d:1 2:c%d:1 3:d%d:1\n", c,a,d,p)
    libc.stdio.fclose(f)

def write_vw_matrix(str target_file,
                 np.ndarray[np.int_t, ndim=1] clicked,
                 np.ndarray[np.int_t, ndim=1] ad_id,
                 np.ndarray[np.int_t, ndim=1] document_id,
                 np.ndarray[np.int_t, ndim=1] platform):
    f = libc.stdio.fopen(target_file.encode('ascii'), 'w')
    cdef int c, a, d, p, i

    for i in range(0, clicked.shape[0]):
        c = clicked[i]
        a = ad_id[i]
        d = document_id[i]
        p = platform[i]
        libc.stdio.fprintf(f, "%d |ad ad%d |doc doc%d |plat plat%d\n", c,a,d,p)
    libc.stdio.fclose(f)

_file_version = 9