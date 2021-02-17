/**
 * LibStruct, original author: Frank Bergmann.
 * Fixes and improvments: Totte Karsson
 */
#pragma hdrstop

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <sstream>
#include <string.h>
#include <vector>
#include <lapacke.h>
#include "lsLibla.h"
#include "lsMatrix.h"
#include "lsUtils.h"

//---------------------------------------------------------------------------

namespace ls {

    using namespace std;
    using namespace ls;

    double gLapackTolerance = 1.0E-12;

    double getTolerance() {
        return gLapackTolerance;
    }

    void setTolerance(double dTolerance) {
        gLapackTolerance = dTolerance;
    }

    vector<Complex> getEigenValues(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getEigenValues " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        vector<Complex> oResult;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();
        lapack_int lwork = 2 * numRows;
        lapack_int info;

        if (numRows != numCols) {
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");
        }

        if (numRows == 0) return oResult;

        lapack_complex_double *eigVals = new lapack_complex_double[numRows];
        memset(eigVals, 0, sizeof(lapack_complex_double) * numRows);

        Matrix<lapack_complex_double> A(oMatrix);

        char job = 'N'; // do not compute eigenvectors
        LAPACKE_zgeev(LAPACK_ROW_MAJOR, job, job, numRows, A.GetPointer(), numRows, eigVals, NULL, numRows, NULL, numRows);


        for (int i = 0; i < numRows; i++) {
            Complex complex(ls::RoundToTolerance(lapack_complex_double_real(eigVals[i]), gLapackTolerance),
                            ls::RoundToTolerance(lapack_complex_double_imag(eigVals[i]), gLapackTolerance));
            oResult.push_back(complex);
        }

        delete[] eigVals;
        return oResult;

    }

    vector<ls::Complex> ZgetEigenValues(ComplexMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== ZgetEigenValues " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        vector<Complex> oResult;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        if (numRows != numCols)
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");

        lapack_complex_double *eigVals = new lapack_complex_double[numRows];
        memset(eigVals, 0, sizeof(lapack_complex_double) * numRows);

        Matrix<lapack_complex_double> A(oMatrix);

        char job = 'N'; // do not compute eigenvectors
        LAPACKE_zgeev(LAPACK_ROW_MAJOR, job, job, numRows, A.GetPointer(), numRows, eigVals, NULL, numRows, NULL, numRows);

        for (int i = 0; i < numRows; i++) {
            Complex complex(ls::RoundToTolerance(lapack_complex_double_real(eigVals[i]), gLapackTolerance),
                            ls::RoundToTolerance(lapack_complex_double_imag(eigVals[i]), gLapackTolerance));
            oResult.push_back(complex);
        }

        delete[] eigVals;

        return oResult;
    }

    vector<DoubleMatrix *> getQRWithPivot(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getQRWithPivot " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;


        vector<DoubleMatrix *> oResult;

        lapack_int row = oMatrix.numRows();
        lapack_int col = oMatrix.numCols();

        if (row * col == 0) {
            DoubleMatrix *oMatrixQ = new DoubleMatrix(row, row);
            DoubleMatrix *oMatrixR = new DoubleMatrix(row, col);
            DoubleMatrix *oMatrixP = new DoubleMatrix(col, col);
            oResult.push_back(oMatrixQ);
            oResult.push_back(oMatrixR);
            oResult.push_back(oMatrixP);
            return oResult;
        }

        lapack_int minRowCol = min(row, col);
        lapack_int lwork = 16 * col;

        double *A = oMatrix.getCopy(true);


        double *Q = NULL;
        if (row * row) {
            Q = new double[row * row];
            memset(Q, 0, sizeof(double) * row * row);
        }
        double *R = NULL;
        if (row * col) {
            R = new double[row * col];
            memset(R, 0, sizeof(double) * row * col);
        }
        double *P = NULL;
        if (col * col) {
            P = new double[col * col];
            memset(P, 0, sizeof(double) * col * col);
        }

        //Log(lDebug5) << "before dorgqr A:\n"    << ls::print(row, col, A);
        //Log(lDebug5) << endl << endl << "Q: \n"    << ls::print(row, row, Q);
        //Log(lDebug5) << endl << endl << "R: \n"    << ls::print(row, col, R);
        //Log(lDebug5) << endl << endl << "P: \n"    << ls::print(col, col, P);

        double *tau = NULL;
        if (minRowCol) {
            tau = new double[minRowCol];
            memset(tau, 0, sizeof(double) * minRowCol);
        }
        lapack_int *jpvt = NULL;
        if (col) {
            jpvt = new lapack_int[col];
            memset(jpvt, 0, sizeof(lapack_int) * col);
        }
        double *work = NULL;
        if (lwork) {
            work = new double[lwork];
            memset(work, 0, lwork);
        }

        int out;

        //Log(lDebug5) << "Input:\n"<<ls::print(row, col, A);

        // call lapack routine dgepq3_ to generate householder reflections
        LAPACKE_dgeqp3_work(LAPACK_ROW_MAJOR, row, col, A, row, jpvt, tau, work, lwork);

        //Log(lDebug5) << "before permutation" << endl;

        // Building permutation matrix P and
        for (int i = 0; i < col; i++) {
            size_t pos = i * col + (jpvt[i] - 1);
            if (pos < col * col)
                P[pos] = 1;
        }

        //Log(lDebug5) << "before memcpy" << endl;

        // set R to A before calling dorgqr
        memcpy(R, A, sizeof(double) * row * col);

        //Log(lDebug5) << "after memcpy" << endl;

        // make R a trapezoidal matrix
        // set Q to A before calling dorgqr
        int index = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < minRowCol; j++) {
                index = i + row * j;
                Q[index] = A[index];
            }

            if (i >= 1)
                for (int j = 0; j < min(i, col); j++) {
                    R[i + row * j] = 0.0;
                }
        }

        //Log(lDebug5) << "before dorgqr:\n"<<ls::print(row, col, A);
        //Log(lDebug5) << endl << endl << "Q: \n"<<ls::print(row, row, Q);
        //Log(lDebug5) << endl << endl << "R: \n"<<ls::print(row, col, R);
        //Log(lDebug5) << endl << endl << "P: \n"<<ls::print(col, col, P);


        // call routine dorgqr_ to build orthogonal matrix Q
        LAPACKE_dorgqr_work(LAPACK_ROW_MAJOR, row, row, minRowCol, Q, row, tau, work, lwork);

        //Log(lDebug5) << endl << endl << "Q: \n"<<ls::print(row, row, Q);
        //Log(lDebug5) << endl << endl << "R: \n"<<ls::print(row, col, R);
        //Log(lDebug5) << endl << endl << "P: \n"<<ls::print(col, col, P);

        DoubleMatrix *oMatrixQ = new DoubleMatrix(Q, row, row, true);
        RoundMatrixToTolerance(*oMatrixQ, gLapackTolerance);

        DoubleMatrix *oMatrixR = new DoubleMatrix(R, row, col, true);
        RoundMatrixToTolerance(*oMatrixR, gLapackTolerance);

        DoubleMatrix *oMatrixP = new DoubleMatrix(P, col, col, true);
        RoundMatrixToTolerance(*oMatrixP, gLapackTolerance);

        oResult.push_back(oMatrixQ);
        oResult.push_back(oMatrixR);
        oResult.push_back(oMatrixP);

        // free
        if (row * col) {
            delete[] A;
        }

        if (row * row) {
            delete[] Q;
        }

        if (row * col) {
            delete[] R;
        }

        if (col * col) {
            delete[] P;
        }

        if (tau) {
            delete[] tau;
        }

        if (jpvt) {
            delete[] jpvt;
        }

        if (work) {
            delete[] work;
        }

        return oResult;
    }

    vector<DoubleMatrix *> getQR(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getQR " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        lapack_int row = oMatrix.numRows();
        lapack_int col = oMatrix.numCols();
        if (row * col == 0) {
            vector<DoubleMatrix *> oResult;
            DoubleMatrix *oMatrixQ = new DoubleMatrix(row, row);
            DoubleMatrix *oMatrixR = new DoubleMatrix(row, col);
            oResult.push_back(oMatrixQ);
            oResult.push_back(oMatrixR);
            return oResult;
        }

        lapack_int minRowCol = min(row, col);

        double *Q = new double[row * row];
        double *R = new double[row * col];
        double *tau = new double[minRowCol];

        double *A = (double *) oMatrix.getCopy(true);


        //Log(lDebug5) << "Input:\n"<<ls::print(row, col, A);

        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, row, col, A, row, tau);

        //Log(lDebug5) << "A: after dgeqrt)\n"<<ls::print(row, col, A);
        //Log(lDebug5) << "tau: after dgeqrt)\n"<<ls::print(1, minRowCol, tau);
        // set R to A before calling dorgqr
        memcpy(R, A, sizeof(double) * row * col);
        int index;
        for (int i = 0; i < row; i++) {
            if (i > 0)
                for (int j = 0; j < min(i, col); j++) {
                    R[i + row * j] = 0.0;
                }
            for (int j = 0; j < minRowCol; j++) {
                index = i + row * j;
                Q[index] = A[index];
            }
        }

        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, row, row, minRowCol, Q, row, tau);

        ls::checkTolerance(row * row, Q, getTolerance());
        ls::checkTolerance(row * col, R, getTolerance());

        //Log(lDebug5) << endl << endl << "Q: )\n"<<ls::print(row, row, Q);
        //Log(lDebug5) << endl << endl << "R: )\n"<<ls::print(row, col, R);

        vector<DoubleMatrix *> oResult;

        DoubleMatrix *oMatrixQ = new DoubleMatrix(Q, row, row, true);
        ls::RoundMatrixToTolerance(*oMatrixQ, gLapackTolerance);

        DoubleMatrix *oMatrixR = new DoubleMatrix(R, row, col, true);
        ls::RoundMatrixToTolerance(*oMatrixR, gLapackTolerance);
        oResult.push_back(oMatrixQ);
        oResult.push_back(oMatrixR);

        delete[] A;
        delete[] Q;
        delete[] R;
        delete[] tau;

        return oResult;
    }


    DoubleMatrix *getLeftNullSpace(DoubleMatrix &oMatrixIn) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getLeftNullSpace " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;
        DoubleMatrix *oMatrix = oMatrixIn.getTranspose();
        DoubleMatrix *oMatrixResult = getRightNullSpace(*oMatrix);
        delete oMatrix;
        //return oMatrixResult;
        DoubleMatrix *oFinalMatrix = oMatrixResult->getTranspose();
        delete oMatrixResult;
        return oFinalMatrix;
    }

    DoubleMatrix *getScaledLeftNullSpace(DoubleMatrix &oMatrixIn) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getScaledLeftNullSpace " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;
        DoubleMatrix *oMatrix = oMatrixIn.getTranspose();
        DoubleMatrix *oMatrixResult = getScaledRightNullSpace(*oMatrix);
        delete oMatrix;
        //return oMatrixResult;
        DoubleMatrix *oFinalMatrix = oMatrixResult->getTranspose();
        delete oMatrixResult;
        return oFinalMatrix;

    }

    DoubleMatrix *getScaledRightNullSpace(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getScaledRightNullSpace " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;
        DoubleMatrix *oTemp = getRightNullSpace(oMatrix);
        DoubleMatrix *oTranspose = oTemp->getTranspose();
        delete oTemp;
        ls::GaussJordan(*oTranspose, gLapackTolerance);
        DoubleMatrix *oResult = oTranspose->getTranspose();
        delete oTranspose;

        ls::RoundMatrixToTolerance(oMatrix, gLapackTolerance);

        return oResult;
    }

    DoubleMatrix *getRightNullSpace(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getRightNullSpace " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;
        DoubleMatrix *oTranspose = oMatrix.getTranspose();

        lapack_int numRows = oTranspose->numRows();
        lapack_int numCols = oTranspose->numCols();

        // determine sizes
        lapack_int min_MN = min(numRows, numCols);
        lapack_int max_MN = max(numRows, numCols);
        lapack_int lwork = 3 * min_MN * min_MN + max(max_MN, 4 * min_MN * min_MN + 4 * min_MN); // 'A'

        // allocate arrays for lapack
        double *A = oTranspose->getCopy(true);
        double *S = new double[min_MN];
        memset(S, 0, sizeof(double) * min_MN);
        double *U = new double[numRows * numRows];
        memset(U, 0, sizeof(double) * numRows * numRows);
        double *VT = new double[numCols * numCols];
        memset(VT, 0, sizeof(double) * numCols * numCols);

        char jobz = 'A';
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, numRows, numCols, A, numRows, S, U, numRows, VT, numCols);

        // now we have everything we could get, now extract the nullspace. In Matlab this would look like:
        //     [U,S,V] = svd(A');
        //     r = rank(A)
        //     Z = U(:,r+1:end)

        int rank = getRank(oMatrix);
        int nResultColumns = numRows - rank;

        DoubleMatrix *oMatrixU = new DoubleMatrix(U, numRows, numRows, true);

        //Log(lDebug5) << " SVD: U " << endl;
        ls::print(*oMatrixU);
        DoubleMatrix *oResultMatrix = new DoubleMatrix(numRows, nResultColumns);
        for (int i = 0; i < nResultColumns; i++) {
            for (int j = 0; j < numRows; j++) {
                (*oResultMatrix)(j, i) = (*oMatrixU)(j, rank + i);
            }
        }
        //Log(lDebug5) << " Nullspace: " << endl;
        ls::print(*oResultMatrix);
        delete[] S;
        delete[] U;
        delete[] VT;
        delete[] A;
        delete oTranspose;
        delete oMatrixU;

        ls::RoundMatrixToTolerance(*oResultMatrix, gLapackTolerance);
        return oResultMatrix;

    }

    int getRank(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getRank " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;
        int rank = 0;
        vector<double> oSingularVals = getSingularValsBySVD(oMatrix);

        for (unsigned int i = 0; i < oSingularVals.size(); i++) {
            if (fabs(oSingularVals[i]) > gLapackTolerance) {
                rank++;
            }
        }
        return rank;
    }

    vector<double> getSingularValsBySVD(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getSingularValsBySVD " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        vector<double> oResult;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        lapack_int min_MN = min(numRows, numCols);
        lapack_int max_MN = max(numRows, numCols);

        if (min_MN == 0) return oResult;

        double *A = oMatrix.getCopy(true);
        double *S = new double[min_MN];
        memset(S, 0, sizeof(double) * min_MN);

        char jobz = 'N';
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, numRows, numCols, A, numRows, S, NULL, numRows, NULL, numCols);

        for (int i = 0; i < min_MN; i++) {
            oResult.push_back(ls::RoundToTolerance(S[i], gLapackTolerance));
        }

        // free memory
        delete[] A;
        delete[] S;

        return oResult;
    }

    ComplexMatrix *getEigenVectors(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getEigenVectors " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        if (numRows != numCols)
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");

        if (numRows == 0) return new ComplexMatrix();

        lapack_complex_double *vr = new lapack_complex_double[numRows * numRows];
        memset(vr, 0, sizeof(lapack_complex_double) * numRows * numRows);

        lapack_complex_double *eigVals = new lapack_complex_double[numRows];
        memset(eigVals, 0, sizeof(lapack_complex_double) * numRows);

        Matrix<lapack_complex_double> A(oMatrix);

        char job = 'N';
        char jobR = 'V'; // compute the right eigenvectors
        LAPACKE_zgeev(LAPACK_ROW_MAJOR, job, jobR, numRows, A.GetPointer(), numRows, eigVals, NULL, numRows, vr, numRows);

        ComplexMatrix *oResult = new ComplexMatrix(numRows, numRows);
        int index;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                index = (j + numRows * i);
                Complex complexNr(
                    ls::RoundToTolerance(lapack_complex_double_real(vr[index]), gLapackTolerance),
                    ls::RoundToTolerance(lapack_complex_double_imag(vr[index]), gLapackTolerance));

                (*oResult)(i, j) = complexNr;//(.set(complex.Real, complex.Imag);
            }
        }

        delete[] eigVals;
        delete[] vr;

        return oResult;
    }

    ComplexMatrix *ZgetEigenVectors(ComplexMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getEigenVectors " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        if (numRows != numCols)
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");

        if (numRows == 0) return new ComplexMatrix();

        lapack_complex_double *vr = new lapack_complex_double[numRows * numRows];
        memset(vr, 0, sizeof(lapack_complex_double) * numRows * numRows);

        lapack_complex_double *eigVals = new lapack_complex_double[numRows];
        memset(eigVals, 0, sizeof(lapack_complex_double) * numRows);

        Matrix<lapack_complex_double> A(oMatrix);

        char job = 'N';
        char jobR = 'V'; // compute the right eigenvectors
        LAPACKE_zgeev(LAPACK_ROW_MAJOR, job, jobR, numRows, A.GetPointer(), numRows, eigVals, NULL, numRows, vr, numRows);

        ComplexMatrix *oResult = new ComplexMatrix(numRows, numRows);
        int index;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                index = (j + numRows * i);
                Complex complexNr(
                    ls::RoundToTolerance(lapack_complex_double_real(vr[index]), gLapackTolerance),
                    ls::RoundToTolerance(lapack_complex_double_imag(vr[index]), gLapackTolerance));

                (*oResult)(i, j) = complexNr;//.set(complex.Real, complex.Imag);
            }
        }

        delete[] eigVals;
        delete[] vr;

        return oResult;
    }

    void
    getSVD(DoubleMatrix &inputMatrix, DoubleMatrix *&outU, std::vector<double> *&outSingularVals, DoubleMatrix *&outV) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getSingularValsBySVD " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        lapack_int numRows = inputMatrix.numRows();
        lapack_int numCols = inputMatrix.numCols();

        lapack_int min_MN = min(numRows, numCols);
        lapack_int max_MN = max(numRows, numCols);

        if (min_MN == 0) return;

        double *A = inputMatrix.getCopy(true);
        double *U = new double[numRows * numRows];
        memset(U, 0, sizeof(double) * numRows * numRows);
        double *VT = new double[numCols * numCols];
        memset(VT, 0, sizeof(double) * numCols * numCols);
        double *S = new double[min_MN];
        memset(S, 0, sizeof(double) * min_MN);


        lapack_int info;
        char jobz = 'A';
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, numRows, numCols, A, numRows, S, U, numRows, VT, numCols);

        outU = new DoubleMatrix(numRows, numRows);
        int index;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                index = (j + numRows * i);
                (*outU)(j, i) = U[index];
            }
        }

        ls::RoundMatrixToTolerance(*outU, gLapackTolerance);

        outV = new DoubleMatrix(numCols, numCols);
        for (int i = 0; i < numCols; i++) {
            for (int j = 0; j < numCols; j++) {
                index = (j + numCols * i);
                (*outV)(i, j) = VT[index];
            }
        }

        ls::RoundMatrixToTolerance(*outV, gLapackTolerance);

        outSingularVals = new std::vector<double>();
        for (int i = 0; i < min_MN; i++) {
            outSingularVals->push_back(ls::RoundToTolerance(S[i], gLapackTolerance));
        }

        // free memory
        delete[] A;
        delete[] S;
        delete[] U;
        delete[] VT;
    }

    void ZgetSVD(ComplexMatrix &inputMatrix, ComplexMatrix *&outU, std::vector<double> *&outSingularVals,
                 ComplexMatrix *&outV) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getSingularValsBySVD " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        lapack_int numRows = inputMatrix.numRows();
        lapack_int numCols = inputMatrix.numCols();

        lapack_int min_MN = min(numRows, numCols);
        lapack_int max_MN = max(numRows, numCols);

        if (min_MN == 0) return;

        lapack_complex_double *A = new lapack_complex_double[numRows * numCols];
        memset(A, 0, sizeof(lapack_complex_double) * numRows * numCols);
        lapack_complex_double *U = new lapack_complex_double[numRows * numRows];
        memset(U, 0, sizeof(lapack_complex_double) * numRows * numRows);
        lapack_complex_double *VT = new lapack_complex_double[numCols * numCols];
        memset(VT, 0, sizeof(lapack_complex_double) * numCols * numCols);
        double *S = new double[min_MN];
        memset(S, 0, sizeof(double) * min_MN);

        int index;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                index = (j + numRows * i);
                A[index]=lapack_make_complex_double(real(inputMatrix(j, i)), imag(inputMatrix(j, i)));
            }
        }

        char jobz = 'A';
        LAPACKE_zgesdd(LAPACK_ROW_MAJOR, jobz, numRows, numCols, A, numRows, S, U, numRows, VT, numCols);

        outU = new ComplexMatrix(numRows, numRows);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                index = (j + numRows * i);
                (*outU)(j, i) = Complex((ls::RoundToTolerance(lapack_complex_double_real(U[index]), gLapackTolerance),
                                         ls::RoundToTolerance(lapack_complex_double_imag(U[index]), gLapackTolerance)));
            }
        }

        outV = new ComplexMatrix(numCols, numCols);
        for (int i = 0; i < numCols; i++) {
            for (int j = 0; j < numCols; j++) {
                index = (j + numCols * i);
                (*outV)(i, j) = Complex(ls::RoundToTolerance(lapack_complex_double_real(VT[index]), gLapackTolerance),
                                        ls::RoundToTolerance(-lapack_complex_double_imag(VT[index]), gLapackTolerance));
            }
        }

        outSingularVals = new std::vector<double>();
        for (int i = 0; i < min_MN; i++) {
            outSingularVals->push_back(ls::RoundToTolerance(S[i], gLapackTolerance));
        }

        // free memory
        delete[] A;
        delete[] S;
        delete[] U;
        delete[] VT;

        return;
    }

    double getRCond(DoubleMatrix &oMatrix) {
        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        lapack_int minRC = min(numRows, numCols);

        if (minRC == 0) {
            return 0.0;
        }

        double *A = (double *) oMatrix.getCopy(true);
        lapack_int *vecP = new lapack_int[minRC];
        memset(vecP, 0, (sizeof(lapack_int) * minRC));

        char norm = '1';
        lapack_int order = numRows * numCols;

        double dnorm = LAPACKE_dlange(LAPACK_ROW_MAJOR, norm, numRows, numCols, A, numRows);
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, numRows, numCols, A, numRows, vecP);
        ls::checkTolerance(numRows * numCols, A, gLapackTolerance);

        double rcond = 0.0;
        LAPACKE_dgecon(LAPACK_ROW_MAJOR, norm, numRows, A, numRows, dnorm, &rcond);

        delete[] vecP;
        delete[] A;
        return rcond;
    }

    LU_Result *getLU(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getLU " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        int minRC = min(numRows, numCols);

        if (minRC == 0) {
            LU_Result *oResult = new LU_Result();
            DoubleMatrix *L = new DoubleMatrix(numRows, minRC);
            DoubleMatrix *U = new DoubleMatrix(minRC, numCols);
            IntMatrix *P = new IntMatrix(numRows, numRows);

            oResult->L = L;
            oResult->U = U;
            oResult->P = P;
            oResult->nInfo = -1;
            return oResult;
        }

        double *A = (double *) oMatrix.getCopy(true);
        lapack_int *vecP = new lapack_int[minRC];
        memset(vecP, 0, (sizeof(lapack_int) * minRC));

        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, numRows, numCols, A, numRows, vecP);

        ls::print(numRows, numCols, A);

        DoubleMatrix *L = new DoubleMatrix(numRows, minRC);
        DoubleMatrix *U = new DoubleMatrix(minRC, numCols);

        // Assign values to Lmat and Umat
        for (int i = 0; i < minRC; i++) {
            (*L)(i, i) = 1.0;
            (*U)(i, i) = A[i + numRows * i];
            for (int j = 0; j < i; j++) {
                (*L)(i, j) = A[i + numRows * j];
            }
            for (int j = i + 1; j < minRC; j++) {
                (*U)(i, j) = A[i + numRows * j];
            }
        }

        if (numRows > numCols) {
            for (int i = numCols; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    (*L)(i, j) = A[i + numRows * j];
                }
            }
        } else {
            for (int i = 0; i < numRows; i++) {
                for (int j = numRows; j < numCols; j++) {
                    (*U)(i, j) = A[i + numRows * j];
                }
            }
        }

        // build permutation matrix
        IntMatrix *P = new IntMatrix(numRows, numRows);
        for (int i = 0; i < numRows; i++)
            (*P)(i, i) = 1;
        for (int i = 0; i < minRC; i++) {
            if (vecP[i] != 0 && vecP[i] - 1 != i)
                P->swapRows(i, vecP[i] - 1);
        }

        LU_Result *oResult = new LU_Result();

        ls::RoundMatrixToTolerance(*L, gLapackTolerance);
        ls::RoundMatrixToTolerance(*U, gLapackTolerance);

        oResult->L = L;
        oResult->U = U;
        oResult->P = P;
        oResult->nInfo = info;

        delete[] A;
        delete[] vecP;

        return oResult;
    }

    LU_Result *getLUwithFullPivoting(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== getLUwithFullPivoting " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        if (numRows != numCols)
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");


        double *A = (double *) oMatrix.getCopy(true);
        lapack_int *vecP = new lapack_int[numRows];
        memset(vecP, 0, (sizeof(lapack_int) * numRows));
        lapack_int *vecQ = new lapack_int[numRows];
        memset(vecQ, 0, (sizeof(lapack_int) * numRows));

        lapack_int info;
        dgetc2_(&numRows, A, &numRows, vecP, vecQ, &info);

        DoubleMatrix *L = new DoubleMatrix(numRows, numRows);
        DoubleMatrix *U = new DoubleMatrix(numRows, numCols);

        // Assign values to Lmat and Umat
        for (int i = 0; i < numRows; i++) {
            (*L)(i, i) = 1.0;
            (*U)(i, i) = A[i + numRows * i];
            for (int j = 0; j < i; j++) {
                (*L)(i, j) = A[i + numRows * j];
            }
            for (int j = i + 1; j < numRows; j++) {
                (*U)(i, j) = A[i + numRows * j];
            }
        }

        if (numRows > numCols) {
            for (int i = numCols; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    (*L)(i, j) = A[i + numRows * j];
                }
            }
        } else {
            for (int i = 0; i < numRows; i++) {
                for (int j = numRows; j < numCols; j++) {
                    (*U)(i, j) = A[i + numRows * j];
                }
            }
        }

        // build permutation matrix
        IntMatrix *P = new IntMatrix(numRows, numRows);
        for (int i = 0; i < numRows; i++)
            (*P)(i, i) = 1;
        for (int i = 0; i < numRows; i++) {
            if (vecP[i] != 0 && vecP[i] - 1 != i)
                P->swapRows(i, vecP[i] - 1);
        }

        IntMatrix *Q = new IntMatrix(numRows, numRows);
        for (int i = 0; i < numRows; i++)
            (*Q)(i, i) = 1;
        for (int i = 0; i < numRows; i++) {
            if (vecQ[i] != 0 && vecQ[i] - 1 != i)
                Q->swapCols(i, vecQ[i] - 1);
        }

        LU_Result *oResult = new LU_Result();

        ls::RoundMatrixToTolerance(*L, gLapackTolerance);
        ls::RoundMatrixToTolerance(*U, gLapackTolerance);

        oResult->L = L;
        oResult->U = U;
        oResult->P = P;
        oResult->Q = Q;
        oResult->nInfo = info;

        delete[] A;
        delete[] vecP;
        delete[] vecQ;

        return oResult;
    }

    DoubleMatrix *inverse(DoubleMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== inverse " << endl;
        //Log(lDebug5) << "======================================================" << endl << endl;
        DoubleMatrix *oResultMatrix = NULL;

        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();


        if (numRows != numCols)
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");


        double *A = oMatrix.getCopy(true);
        lapack_int *ipvt = new lapack_int[numRows];
        memset(ipvt, 0, sizeof(lapack_int) * numRows);

        //Log(lDebug5) << "Input Matrix 1D: \n"<<ls::print(numRows, numRows, A);


        // Carry out LU Factorization
        lapack_int info;
        info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, numRows, numRows, A, numRows, ipvt);

        if (info < 0)
            throw ApplicationException("Error in dgetrf : LU Factorization", "Illegal Value");
        if (info > 0)
            throw ApplicationException("Exception in ls while computing Inverse", "Input Matrix is Singular.");


        //Log(lDebug5) << "After dgetrf: \n"<<ls::print(numRows, numRows, A);

        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, numRows, A, numRows, ipvt);

        //Log(lDebug5) << "After dgetri: \n"<<ls::print(numRows, numRows, A);

        oResultMatrix = new DoubleMatrix(A, numRows, numRows, true);
        ls::RoundMatrixToTolerance(*oResultMatrix, gLapackTolerance);

        delete[] A;
        delete[] ipvt;

        return oResultMatrix;
    }

    ComplexMatrix *Zinverse(ComplexMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== Zinverse " << endl;
        //Log(lDebug5) << "======================================================" << endl;

        ComplexMatrix *oResultMatrix = NULL;
        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        if (numRows != numCols) {
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");
        }

        lapack_complex_double *A = new lapack_complex_double[numRows * numRows];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                A[i + numRows * j] = lapack_make_complex_double(real(oMatrix(i, j)), imag(oMatrix(i, j)));
            }
        }

        ////Log(lDebug5) << "Input Matrix 1D: \n";
        //ls::print(numRows, numRows, A);

        lapack_int *ipvt = new lapack_int[numRows];
        memset(ipvt, 0, sizeof(lapack_int) * numRows);

        // Carry out LU Factorization
        lapack_int info;
        info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, numRows, numRows, A, numRows, ipvt);

        if (info < 0) {
            throw ApplicationException("Error in dgetrf : LU Factorization", "Illegal Value");
        }

        if (info > 0) {
            throw ApplicationException("Exception in ls while computing Inverse", "Input Matrix is Sinuglar.");
        }

        ////Log(lDebug5) << "After dgetrf: \n";
        //ls::print(numRows, numRows, A);

        info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, numRows, A, numRows, ipvt);

        //Log(lDebug5) << "After dgetri: \n";
        //ls::print(numRows, numRows, A);

        oResultMatrix = new ComplexMatrix(numRows, numRows);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                Complex tols(ls::RoundToTolerance(lapack_complex_double_real(A[(i + numRows * j)]), gLapackTolerance),
                             ls::RoundToTolerance(lapack_complex_double_imag(A[(i + numRows * j)]), gLapackTolerance));
                (*oResultMatrix)(i, j) = tols;

            }
        }

        // free memory
        delete[] A;
        delete[] ipvt;

        return oResultMatrix;
    }

    ComplexMatrix *Zinverse(const ComplexMatrix &oMatrix) {
        //Log(lDebug5) << "======================================================" << endl;
        //Log(lDebug5) << "=== Zinverse " << endl;
        //Log(lDebug5) << "======================================================" << endl;

        ComplexMatrix *oResultMatrix = NULL;
        lapack_int numRows = oMatrix.numRows();
        lapack_int numCols = oMatrix.numCols();

        if (numRows != numCols) {
            throw ApplicationException("Input Matrix must be square", "Expecting a Square Matrix");
        }

        lapack_complex_double *A = new lapack_complex_double[numRows * numRows];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                A[i + numRows * j] = lapack_make_complex_double(real(oMatrix(i, j)), imag(oMatrix(i, j)));
            }
        }

        ////Log(lDebug5) << "Input Matrix 1D: \n";
        //ls::print(numRows, numRows, A);

        lapack_int *ipvt = new lapack_int[numRows];
        memset(ipvt, 0, sizeof(lapack_int) * numRows);

        // Carry out LU Factorization
        lapack_int info;
        info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, numRows, numRows, A, numRows, ipvt);

        if (info < 0) {
            throw ApplicationException("Error in dgetrf : LU Factorization", "Illegal Value");
        }

        if (info > 0) {
            throw ApplicationException("Exception in ls while computing Inverse", "Input Matrix is Sinuglar.");
        }

        ////Log(lDebug5) << "After dgetrf: \n";
        //ls::print(numRows, numRows, A);

        info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, numRows, A, numRows, ipvt);

        //Log(lDebug5) << "After dgetri: \n";
        //ls::print(numRows, numRows, A);

        oResultMatrix = new ComplexMatrix(numRows, numRows);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numRows; j++) {
                Complex tols(ls::RoundToTolerance(lapack_complex_double_real(A[(i + numRows * j)]), gLapackTolerance),
                             ls::RoundToTolerance(lapack_complex_double_imag(A[(i + numRows * j)]), gLapackTolerance));
                (*oResultMatrix)(i, j) = tols;
            }
        }

        // free memory
        delete[] A;
        delete[] ipvt;
        return oResultMatrix;
    }



//LibLA* getInstance()
//{
//    if (_Instance == NULL)
//        _Instance = new LibLA();
//    return _Instance;
//}


// static variable definitions
//LibLA* _Instance = NULL;


} //Namespace ls
