cimport libc.stdio
cimport libc.stdlib
import nose
import numpy as np
cimport numpy as np
cimport cython

# Quickly write to a LIBFFM compatible file.
# This saves *masses* of time by looping in C
# and using raw C io.
@cython.boundscheck(False)
def write_ffm_matrix(str target_file,
                     np.ndarray[np.int_t, ndim=1] clicked,
                     np.ndarray[np.int_t, ndim=1] ad_id,
                     np.ndarray[np.int_t, ndim=1] document_id,
                     np.ndarray[np.int_t, ndim=1] platform):
    cdef int c, a, d, p, i
    f = libc.stdio.fopen(target_file.encode('ascii'), 'w')

    nose.tools.assert_equal(clicked.shape[0], ad_id.shape[0])
    nose.tools.assert_equal(clicked.shape[0], document_id.shape[0])
    nose.tools.assert_equal(clicked.shape[0], platform.shape[0])
    for i in range(0, clicked.shape[0]):
        c = clicked[i]
        a = ad_id[i]
        d = document_id[i]
        p = platform[i]
        libc.stdio.fprintf(f, "%d 0:%d:1 1:%d:1 2:%d:1 3:%d:1\n", c,a,d,p)

    libc.stdio.fclose(f)

@cython.boundscheck(False)
def write_vw_matrix(str target_file,
                    np.ndarray[np.int_t, ndim=1] clicked,
                    np.ndarray[np.int_t, ndim=1] ad_id,
                    np.ndarray[np.int_t, ndim=1] document_id,
                    np.ndarray[np.int_t, ndim=1] platform,
                    np.ndarray[np.int_t, ndim=1] user_id,
                    np.ndarray[np.int_t, ndim=1] country_id,
                    np.ndarray[np.int_t, ndim=1] state_id,
                    ):
    cdef int c, a, d, p, i, u, co, st
    f = libc.stdio.fopen(target_file.encode('ascii'), 'w')

    nose.tools.assert_equal(clicked.shape[0], ad_id.shape[0])
    nose.tools.assert_equal(clicked.shape[0], document_id.shape[0])
    nose.tools.assert_equal(clicked.shape[0], user_id.shape[0])
    nose.tools.assert_equal(clicked.shape[0], platform.shape[0])
    nose.tools.assert_equal(clicked.shape[0], country_id.shape[0])
    nose.tools.assert_equal(clicked.shape[0], state_id.shape[0])
    for i in range(0, clicked.shape[0]):
        c = 1 if  clicked[i] else -1
        a = ad_id[i]
        d = document_id[i]
        u = user_id[i]
        p = platform[i]
        co = country_id[i]
        st = state_id[i]
        libc.stdio.fprintf(f, "%d |ad ad%d |doc doc%d |plat plat%d |user user%d |geoC cou%d st%d\n", c,a,d,p,u,co,st)
    libc.stdio.fclose(f)

_file_version = 15