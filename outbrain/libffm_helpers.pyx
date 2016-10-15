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
                 np.ndarray[np.int_t, ndim=1] platform,
                 np.ndarray[np.int_t, ndim=1] uuid):
    f = libc.stdio.fopen(target_file.encode('ascii'), 'w')
    cdef int c, a, d, p, u, i

    for i in range(0, clicked.shape[0]):
        c = clicked[i]
        a = ad_id[i]
        d = document_id[i]
        p = platform[i]
        u = uuid[i]
        libc.stdio.fprintf(f, "%d 0:%d:1 1:%d:1 2:%d:1 3:%d:1 4:%d:1\n", c,a,d,p,u)
    libc.stdio.fclose(f)

def write_vw_matrix(str target_file,
                 np.ndarray[np.int_t, ndim=1] clicked,
                 np.ndarray[np.int_t, ndim=1] ad_id,
                 np.ndarray[np.int_t, ndim=1] document_id,
                 np.ndarray[np.int_t, ndim=1] platform,
                 np.ndarray[np.int_t, ndim=1] uuid):
    f = libc.stdio.fopen(target_file.encode('ascii'), 'w')
    cdef int c, a, d, p, u, i

    for i in range(0, clicked.shape[0]):
        c = clicked[i]
        a = ad_id[i]
        d = document_id[i]
        p = platform[i]
        u = uuid[i]
        libc.stdio.fprintf(f, "%d |ad %d:1 |doc %d:1 |plat %d:1 |user %d:1\n", c,a,d,p,u)
    libc.stdio.fclose(f)

_file_version = 3
