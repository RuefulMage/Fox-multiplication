#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <cmath>
int ProcNum = 0;
int ProcRank = 0;
int GridSize;

int GridCoords[2];
MPI_Comm GridComm;
MPI_Comm ColComm;
MPI_Comm RowComm;

void RandomDataInitialization
        (double* matrixA, double* matrixB,
         int Size) {
    int i, j;
    srand(unsigned(clock()));
    for (i = 0; i<Size; i++)
        for (j = 0; j<Size; j++) {
            matrixA[i*Size + j] = rand() / double(1000);
            matrixB[i*Size + j] = rand() / double(1000);
        }
}

void PrintMatrix
        (double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i = 0; i<RowCount; i++) {
        for (j = 0; j<ColCount; j++)
            printf("%7.4f ", pMatrix[i*ColCount + j]);
        printf("\n");
    }
}

void SerialResultCalculation
        (double* matrixA, double* matrixB,
         double* pCMatrix, int Size) {
    int i, j, k;  // Loop variables
    for (i = 0; i<Size; i++) {
        for (j = 0; j<Size; j++)
            for (k = 0; k<Size; k++)
                pCMatrix[i*Size + j] += matrixA[i*Size + k] * matrixB[k*Size + j];
    }
}

void BlockMultiplication
        (double* pAblock, double* pBblock,
         double* pCblock, int Size) {
    SerialResultCalculation(pAblock, pBblock, pCblock, Size);
}


void CreateGridCommunicators
        () {
    int DimSize[2];
    int Periodic[2];
    int Subdims[2];
    DimSize[0] = GridSize;

    DimSize[1] = GridSize;
    Periodic[0] = 0;
    Periodic[1] = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);

    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

    Subdims[0] = 0;
    Subdims[1] = 1;
    MPI_Cart_sub(GridComm, Subdims, &RowComm);

    Subdims[0] = 1;
    Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}

void ProcessInitialization
        (double* &matrixA, double* &matrixB,
         double* &pCMatrix, double* &pAblock, double* &pBblock, double* &pCblock,
         double* &pTemporaryAblock, int &Size, int &BlockSize) {
    if (ProcRank == 0) {
        do {
            printf("\nEnter the size of matrices: ");
            scanf("%d", &Size);
            if (Size%GridSize != 0) {
                printf("Size of matrices must be divisible by the grid size!\n");
            }
        } while (Size%GridSize != 0);
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    BlockSize = Size / GridSize;
    pAblock = new double[BlockSize*BlockSize];
    pBblock = new double[BlockSize*BlockSize];
    pCblock = new double[BlockSize*BlockSize];
    pTemporaryAblock = new double[BlockSize*BlockSize];
    for (int i = 0; i<BlockSize*BlockSize; i++) {
        pCblock[i] = 0;
    }
    if (ProcRank == 0) {
        matrixA = new double[Size*Size];
        matrixB = new double[Size*Size];
        pCMatrix = new double[Size*Size];
        RandomDataInitialization(matrixA, matrixB, Size);

        printf("\n*******************************************\n");
        printf("The resul of initialization of MatrixA\n");
        for (int i = 0; i < Size; i++) {
            printf("\n");
            for (int j = 0; j < Size; j++) {
                printf("%lf ", matrixA[i * Size + j]);
            }
        }
        printf("\n*******************************************\n");
        printf("The resul of initialization of MatrixB\n");
        for (int i = 0; i < Size; i++) {
            printf("\n");
            for (int j = 0; j < Size; j++) {
                printf("%lf ", matrixB[i * Size + j]);
            }
        }
    }
}

void CheckerboardMatrixScatter
        (double* pMatrix, double* pMatrixBlock,
         int Size, int BlockSize) {
    double * MatrixRow = new double[BlockSize*Size];
    if (GridCoords[1] == 0) {
        MPI_Scatter(pMatrix, BlockSize*Size, MPI_DOUBLE, MatrixRow,
                    BlockSize*Size, MPI_DOUBLE, 0, ColComm);
    }
    for (int i = 0; i<BlockSize; i++) {
        MPI_Scatter(&MatrixRow[i*Size], BlockSize, MPI_DOUBLE,
                    &(pMatrixBlock[i*BlockSize]), BlockSize, MPI_DOUBLE, 0, RowComm);
    }
    delete[] MatrixRow;
}

void DataDistribution
        (double* matrixA, double* matrixB, double*
        pMatrixAblock, double* pBblock, int Size, int BlockSize) {
    CheckerboardMatrixScatter(matrixA, pMatrixAblock, Size, BlockSize);
    CheckerboardMatrixScatter(matrixB, pBblock, Size, BlockSize);
}

void ResultCollection
        (double* pCMatrix, double* pCblock, int Size,
         int BlockSize) {
    double * pResultRow = new double[Size*BlockSize];
    for (int i = 0; i<BlockSize; i++) {
        MPI_Gather(&pCblock[i*BlockSize], BlockSize, MPI_DOUBLE,
                   &pResultRow[i*Size], BlockSize, MPI_DOUBLE, 0, RowComm);
    }
    if (GridCoords[1] == 0) {
        MPI_Gather(pResultRow, BlockSize*Size, MPI_DOUBLE, pCMatrix,
                   BlockSize*Size, MPI_DOUBLE, 0, ColComm);
    }
    delete[] pResultRow;
}

void ABlockCommunication
        (int iter, double *pAblock, double* pMatrixAblock,
         int BlockSize) {

    int Pivot = (GridCoords[0] + iter) % GridSize;
    if (GridCoords[1] == Pivot) {
        for (int i = 0; i<BlockSize*BlockSize; i++)
            pAblock[i] = pMatrixAblock[i];
    }

    MPI_Bcast(pAblock, BlockSize*BlockSize, MPI_DOUBLE, Pivot, RowComm);
}

void BblockCommunication
        (double *pBblock, int BlockSize) {
    MPI_Status Status;
    int NextProc = GridCoords[0] + 1;
    if (GridCoords[0] == GridSize - 1) NextProc = 0;
    int PrevProc = GridCoords[0] - 1;
    if (GridCoords[0] == 0) PrevProc = GridSize - 1;
    MPI_Sendrecv_replace(pBblock, BlockSize*BlockSize, MPI_DOUBLE,
                         NextProc, 0, PrevProc, 0, ColComm, &Status);
}



void ParallelResultCalculation
        (double* pAblock, double* pMatrixAblock,
         double* pBblock, double* pCblock, int BlockSize) {
    for (int iter = 0; iter < GridSize; iter++) {

        ABlockCommunication(iter, pAblock, pMatrixAblock, BlockSize);

        BlockMultiplication(pAblock, pBblock, pCblock, BlockSize);

        BblockCommunication(pBblock, BlockSize);
    }
}
//void TestBlocks
//        (double* pBlock, int BlockSize, char str[]) {
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (ProcRank == 0) {
//        printf("%s \n", str);
//    }
//    for (int i = 0; i<ProcNum; i++) {
//        if (ProcRank == i) {
//            printf("ProcRank = %d \n", ProcRank);
//            PrintMatrix(pBlock, BlockSize, BlockSize);
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
//}

void TestResult
        (double* matrixA, double* matrixB, double* pCMatrix,
         int Size) {
    double* pSerialResult;
    double Accuracy = 1.e-6;
    int equal = 0;
    int i;
    if (ProcRank == 0) {
        pSerialResult = new double[Size*Size];
        for (i = 0; i<Size*Size; i++) {
            pSerialResult[i] = 0;
        }

        BlockMultiplication(matrixA, matrixB, pSerialResult, Size);
        for (i = 0; i<Size*Size; i++) {
            if (fabs(pSerialResult[i] - pCMatrix[i]) >= Accuracy)
                equal = 1;
        }

        printf("\n*******************************************\n");
        printf("Output the resul of serial algorithm\n");
        for (int i = 0; i < Size; i++) {
            printf("\n");
            for (int j = 0; j < Size; j++) {
                printf("%lf ", pSerialResult[i * Size + j]);
            }
        }
        printf("\n*******************************************\n");
        printf("Output the resul of parallel algorithm\n");
        for (int i = 0; i < Size; i++) {
            printf("\n");
            for (int j = 0; j < Size; j++) {
                printf("%lf ", pCMatrix[i * Size + j]);
            }
        }
        printf("\n*******************************************\n");
        if (equal == 1)
            printf(" \nThe results of serial and parallel andlgorithms are NOT identical. Check your code.");
        else
            printf(" \nThe results of serial and parallel algorithms are identical.");
    }
}

void ProcessTermination
        (double* matrixA, double* matrixB,
         double* pCMatrix, double* pAblock, double* pBblock, double* pCblock,
         double* pMatrixAblock) {
    if (ProcRank == 0) {
        delete[] matrixA;
        delete[] matrixB;
        delete[] pCMatrix;
    }
    delete[] pAblock;
    delete[] pBblock;
    delete[] pCblock;
    delete[] pMatrixAblock;
}

using namespace std;

int main(int argc, char* argv[]) {
    double* matrixA;
    double* matrixB;
    double* pCMatrix;
    int Size;
    int BlockSize;
    double *pAblock;
    double *pBblock;
    double *pCblock;
    double *pMatrixAblock;
    double Start, Finish, Duration;
    setvbuf(stdout, 0, _IONBF, 0);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    GridSize = sqrt((double)ProcNum);
    if (ProcNum != GridSize*GridSize) {
        if (ProcRank == 0) {
            printf("Number of processes must be a perfect square \n");
        }
    }
    else {
        if (ProcRank == 0)
            printf("Parallel matrix multiplication program\n");

        CreateGridCommunicators();

        ProcessInitialization(matrixA, matrixB, pCMatrix, pAblock, pBblock,
                              pCblock, pMatrixAblock, Size, BlockSize);
        DataDistribution(matrixA, matrixB, pMatrixAblock, pBblock, Size,
                         BlockSize);

        ParallelResultCalculation(pAblock, pMatrixAblock, pBblock,
                                  pCblock, BlockSize);

        ResultCollection(pCMatrix, pCblock, Size, BlockSize);
        TestResult(matrixA, matrixB, pCMatrix, Size);

        ProcessTermination(matrixA, matrixB, pCMatrix, pAblock, pBblock,
                           pCblock, pMatrixAblock);
    }
    int tmp;
//    printf("\nInput something to EXIT: ");
//    scanf("%d", &tmp);
    MPI_Finalize();
}

