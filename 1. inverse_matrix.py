# @source: https://github.com/lihiSabag/Numerical-Analysis-2023.git

from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
import numpy as np

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""

def inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    # 1 = col , 0 = row
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    if not np.linalg.det(matrix):
        raise ValueError("Matrix is singular, cannot find its inverse.")
    # Creating an Identity Matrix of the Same Size
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            """raise ValueError("Matrix is singular, cannot find its inverse.")"""
            pivot_row = i
            v_max = 0
            for j in range(i+1, n):
                if abs(matrix[j][i]) > v_max:
                    v_max = abs(matrix[j][i])
                    pivot_row = j

            if not matrix[pivot_row][i]:  # Checking if the diagonal element is zero
                return i  # Matrix is singular

                # Swap the current row with the pivot row
            if pivot_row != i:
                # Swap entire rows, including the augmented column
                SaveRowi = matrix[i].copy()
                matrix[i] = matrix[pivot_row]
                matrix[pivot_row] = SaveRowi
                SaveRowi = identity[i].copy()
                identity[i] = identity[pivot_row]
                identity[pivot_row] = SaveRowi

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            """print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")"""
            matrix = np.dot(elementary_matrix, matrix)
            """print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)"""
            identity = np.dot(elementary_matrix, identity)

    # Zero out the elements
        for j in range(n):
            if i < j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)
    # Zero out the elements
    for i in range(n - 1, -1, -1):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")
        for j in range(n-1, -1, -1):
            if i > j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j + 1} = R{j + 1} + ({scalar}R{i + 1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN,
                      "------------------------------------------------------------------------------------------------------------------",
                        bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)
    for k in range(n):
        for w in range(n):
            if abs(matrix[k][w]) < 1e-10:  # Small values are treated as zeros
                matrix[k][w] = 0
            if abs(identity[k][w]) < 1e-10:  # Small values are treated as zeros
                identity[k][w] = 0
    return identity

if __name__ == '__main__':

    """A = np.array([[0, 1,  2],
                [3, 4,  5],
                [6,  7, 8]])"""
    A = np.array([
        [-1, 1, 3, -3, 1],
        [3, -3, -4, 2, 3],
        [2, 1, -5, -3, 5],
        [-5, -6, 4, 1, 3],
        [3, -2, -2, -3, 5]])

    """A = np.array([  [2, 1, 0],
                    [3, -1, 0],
                    [1, 4, -2]])"""

    try:
        A_inverse = inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print(
            "=====================================================================================================================",
            bcolors.ENDC)
        # Check for me to see if this is correct
        ans = np.dot(A_inverse, A)
        n = len(A)
        for k in range(n):
            for w in range(n):
                if abs(ans[k][w]) < 1e-10:  # Small values are treated as zeros
                    ans[k][w] = 0
        print(bcolors.OKBLUE, "\nans A * A*-1: \n", ans)
    except ValueError as e:
        print(str(e))
