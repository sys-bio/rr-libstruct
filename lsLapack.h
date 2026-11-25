#ifndef LS_LAPACK_H_
#define LS_LAPACK_H_

/*
 * Declarations of the BLAS and LAPACK routines that rr-libstruct and roadrunner call, as exported
 * by OpenBLAS. The routines use the Fortran calling convention: every argument is passed by
 * pointer and matrices are stored column-major. OpenBLAS must be built with 32-bit integers
 * (no INTERFACE64).
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef int ls_int;
typedef struct { double r, i; } ls_dcomplex;

/* BLAS */
void dcopy_(ls_int *n, double *x, ls_int *incx, double *y, ls_int *incy);
void daxpy_(ls_int *n, double *alpha, double *x, ls_int *incx, double *y, ls_int *incy);
double dnrm2_(ls_int *n, double *x, ls_int *incx);

/* LU factorization and inverse */
void dgetrf_(ls_int *m, ls_int *n, double *a, ls_int *lda, ls_int *ipiv, ls_int *info);
void dgetri_(ls_int *n, double *a, ls_int *lda, ls_int *ipiv, double *work, ls_int *lwork, ls_int *info);
void dgetc2_(ls_int *n, double *a, ls_int *lda, ls_int *ipiv, ls_int *jpiv, ls_int *info);
void zgetrf_(ls_int *m, ls_int *n, ls_dcomplex *a, ls_int *lda, ls_int *ipiv, ls_int *info);
void zgetri_(ls_int *n, ls_dcomplex *a, ls_int *lda, ls_int *ipiv, ls_dcomplex *work, ls_int *lwork, ls_int *info);

/* matrix norm and condition number */
double dlange_(char *norm, ls_int *m, ls_int *n, double *a, ls_int *lda, double *work);
void dgecon_(char *norm, ls_int *n, double *a, ls_int *lda, double *anorm, double *rcond, double *work, ls_int *iwork, ls_int *info);

/* QR factorization */
void dgeqp3_(ls_int *m, ls_int *n, double *a, ls_int *lda, ls_int *jpvt, double *tau, double *work, ls_int *lwork, ls_int *info);
void dgeqrf_(ls_int *m, ls_int *n, double *a, ls_int *lda, double *tau, double *work, ls_int *lwork, ls_int *info);
void dorgqr_(ls_int *m, ls_int *n, ls_int *k, double *a, ls_int *lda, double *tau, double *work, ls_int *lwork, ls_int *info);

/* singular value decomposition */
void dgesvd_(char *jobu, char *jobvt, ls_int *m, ls_int *n, double *a, ls_int *lda, double *s, double *u, ls_int *ldu, double *vt, ls_int *ldvt, double *work, ls_int *lwork, ls_int *info);
void dgesdd_(char *jobz, ls_int *m, ls_int *n, double *a, ls_int *lda, double *s, double *u, ls_int *ldu, double *vt, ls_int *ldvt, double *work, ls_int *lwork, ls_int *iwork, ls_int *info);
void zgesdd_(char *jobz, ls_int *m, ls_int *n, ls_dcomplex *a, ls_int *lda, double *s, ls_dcomplex *u, ls_int *ldu, ls_dcomplex *vt, ls_int *ldvt, ls_dcomplex *work, ls_int *lwork, double *rwork, ls_int *iwork, ls_int *info);

/* eigenvalues */
void zgeev_(char *jobvl, char *jobvr, ls_int *n, ls_dcomplex *a, ls_int *lda, ls_dcomplex *w, ls_dcomplex *vl, ls_int *ldvl, ls_dcomplex *vr, ls_int *ldvr, ls_dcomplex *work, ls_int *lwork, double *rwork, ls_int *info);

#ifdef __cplusplus
}
#endif

#endif
