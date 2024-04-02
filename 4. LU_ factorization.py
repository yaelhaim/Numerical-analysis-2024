# @source: https://github.com/lihiSabag/Numerical-Analysis-2023.git
import numpy as np
from colors import bcolors
from matrix_utility import swap_rows_elementary_matrix, row_addition_elementary_matrix, MultiplyMatrix

"""
Function lu(A):
    Input: A - a square matrix
    Output: L, U - the lower and upper triangular matrices of A

    N = size of A
    L = identity matrix of size N x N

    For i from 0 to N-1:
        Find pivot row with largest absolute value in current column
        If pivot element is 0:
            Raise ValueError("matrix is singular")

        If pivot row is not equal to i:
            Create elementary matrix to swap rows
            Apply elementary matrix to A
            Apply inverse of elementary matrix to L

        For j from i+1 to N-1:
            Compute multiplier
            Create elementary matrix to zero out element below pivot
            Apply elementary matrix to A
            Apply inverse of elementary matrix to L

    Check if diagonal elements of U are zero
    If any diagonal element of U is zero:
        Raise ValueError("matrix is singular")

    Return L, U

Function backward_substitution(mat):
    Input: mat - upper triangular matrix augmented with right-hand side vector
    Output: x - solution vector

    N = size of mat
    Create an array x to store solution
    
    For i from N-1 down to 0:
        Set x[i] as the last column element of mat[i]
        For j from i+1 to N-1:
            Subtract mat[i][j] multiplied by x[j] from x[i]
        Divide x[i] by mat[i][i]

    Return x

Function lu_solve(A_b):
    Input: A_b - augmented matrix [A | b] where A is a square matrix and b is the right-hand side vector
    Output: Prints the solution vector if successful, otherwise prints an error message

    Try:
        Compute LU decomposition of A_b to obtain L and U
        Print Lower triangular matrix L and Upper triangular matrix U
        
        Compute B = L * U
        Print B
        
        Compute solution vector using backward substitution with U
        Print solution vector
        
    Except ValueError as e:
        Print the error message


"""

def lu(A):
        N = len(A)
        L = np.eye(N)  # Create an identity matrix of size N x N
        for i in range(N):
            # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
            pivot_row = i
            v_max = abs(A[pivot_row][i])
            for j in range(i + 1, N):
                if abs(A[j][i]) > v_max:
                    v_max = abs(A[j][i])
                    pivot_row = j

            # if a principal diagonal element is zero,it denotes that matrix is singular,
            # and will lead to a division-by-zero later.
            """if A[i][pivot_row] == 0:"""
            if A[pivot_row][i] == 0:
                raise ValueError("matrix is singular")

            # Swap the current row with the pivot row
            if pivot_row != i:
                e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
                print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
                A = np.matmul(e_matrix, A)
                print(f"The matrix after elementary operation")
                print(np.array(A))
                print(bcolors.OKGREEN,"---------------------------------------------------------------------------", bcolors.ENDC)

            for j in range(i + 1, N):
                #  Compute the multiplier
                if A[i][i] == 0:
                    raise ValueError("matrix is singular")
                m = -A[j][i] / A[i][i]
                e_matrix = row_addition_elementary_matrix(N, j, i, m)
                e_inverse = np.linalg.inv(e_matrix)
                L = np.matmul(L, e_inverse)
                A = np.matmul(e_matrix, A)
                print(f"elementary matrix to zero the element in row {j} below the pivot in column {i} :\n {e_matrix} \n")
                print(f"The matrix after elementary operation")
                print(np.array(A))
                print(bcolors.OKGREEN, "---------------------------------------------------------------------------",
                      bcolors.ENDC)
        # Checking if the coordinates are zeros
        for i in range(N):
            if not A[i][i]:
                raise ValueError("matrix is singular")
        u = A
        return L, u

# function to calculate the values of the unknowns
def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution
    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):
        x[i] = mat[i][N]
        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]
        x[i] = (x[i] / mat[i][i])
    return x
def lu_solve(A_b):
    try:
        L, U = lu(A_b)
        print("Lower triangular matrix L:\n", np.array(L))
        print("Upper triangular matrix U:\n", np.array(U))
        B = np.matmul(L, U)
        print("\n", B)

        # The segment that computes the solutions of the matrix
        result = backward_substitution(U)
        print(bcolors.OKBLUE, "\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))

    except ValueError as e:
        print(str(e))
if __name__ == '__main__':
    A_b = [ [1,    4,  -3, 11],
            [-2,  8,    5, 25],
            [3,    4,  7,  25]]
    """A_b = [
        [2, 4, -4],
        [1, -4, 3],
        [-6, -9, 5]]"""
    lu_solve(np.array(A_b))
