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


def write_lightgbm(str target_file,
                   np.ndarray[np.int_t, ndim=1] clicked,
                   np.ndarray[np.int_t, ndim=2] discrete_vals,
                   np.ndarray[np.float32_t, ndim=2] continuous_vals):
    f = libc.stdio.fopen(target_file.encode('ascii'), 'w')
    cdef int num_rows, num_d_cols, num_c_cols
    cdef int discrete_val, i, j
    cdef float cont_val
    num_rows = clicked.shape[0]
    num_d_cols = discrete_vals.shape[1]
    num_c_cols = continuous_vals.shape[1]

    nose.tools.assert_equal(num_rows, discrete_vals.shape[0])
    nose.tools.assert_equal(num_rows, continuous_vals.shape[0])

    for i in range(0, clicked.shape[0]):
        discrete_val = clicked[i]
        libc.stdio.fprintf(f, "%d ", discrete_val)
        for j in range(0, num_d_cols):
            discrete_val = discrete_vals[i, j]
            libc.stdio.fprintf(f, "%d:%d ", j, discrete_val)

        for j in range(0, num_c_cols):
            cont_val = continuous_vals[i, j]
            libc.stdio.fprintf(f, "%d:%d ", j+num_d_cols, cont_val)
        libc.stdio.fprintf(f, "\n")
    libc.stdio.fclose(f)

_file_version = 16
