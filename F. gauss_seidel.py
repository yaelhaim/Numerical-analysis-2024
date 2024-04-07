"""
Input: Matrix A of size n×n, pointing to vector b of length n, initial guess X0, tolerance TOL (default 1e-16), maximum number of iterations N (default 200).

1. Check if A produces a matrix with a dominant diagonal. If not, raise an error.
2. Initialize variable k to 1.
3. Print a message indicating that the matrix has a dominant diagonal and Gauss-Seidel algorithm will be used.
4. Print a header row containing the column names x1 to xn for the table to be printed.
5. Loop to perform the algorithm while k is less than or equal to N:
   - For each i from 0 to n-1:
     * Compute sigma, the sum of all variables excluding the current one (x_j) in row i, such that j ≠ i:
       sigma = Σ (A[i][j] * x[j]) for j ≠ i
     * Update the value of variable x[i] using the formula:
       x[i] = (b[i] - sigma) / A[i][i]
   - Print the iteration number k and the values of variables x1 to xn.
   - Check if the difference between X0 and x is less than the tolerance TOL using the infinity-norm distance measurement. If yes, return the values of the variables.
   - Update the value of k.
   - Update the variable X0 to be equal to x.
6. Print an error message if the maximum number of iterations N is exceeded.
7. Return the values of variables x1 to xn if it exits the loop.

"""

import numpy as np
from numpy.linalg import norm

from colors import bcolors
from matrix_utility import is_diagonally_dominant


def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1
    print('preforming gauss seidel algorithm\n')
    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)
        k += 1
        X0 = x.copy()
    print("Maximum number of iterations exceeded")
    return tuple(x)

if __name__ == '__main__':

    A = np.array([[9, 3, 1], [4, 2, 1], [1, 1, 1]])
    b = np.array([-1, 4, 3])
    X0 = np.zeros_like(b)

    try:
        solution =gauss_seidel(A, b, X0)
        solution = tuple(map(lambda x: round(x, 2), solution))
        print(bcolors.OKBLUE,"\nApproximate solution:", solution)

    except ValueError as e:
        print(str(e))
