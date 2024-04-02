# @source: https://github.com/lihiSabag/Numerical-Analysis-2023.git
import numpy as np
from Matrix.inverse_matrix import inverse
from colors import bcolors
from matrix_utility import print_matrix
"""
Function norm(mat):
    Input: mat - square matrix
    Output: max_norm - maximum norm of the matrix
    
    size = length of mat
    max_row = 0  # This variable will hold the maximum sum of values in a row
    
    For each row in the matrix:
        sum_row = 0  # Initialize the sum of values in the row
        For each column in the matrix:
            sum_row += absolute value of the value at [row][column]
        If sum_row is greater than max_row:
            max_row = sum_row  # Update the maximum row norm
        
    Return max_row  # Return the maximum row norm

Function condition_number(A):
    Input: A - matrix
    Output: cond - condition number of matrix A
    
    norm_A = norm(A)  # Calculate the maximum norm (infinity norm) of matrix A
    A_inv = inverse(A)  # Calculate the inverse of matrix A
    norm_A_inv = norm(A_inv)  # Calculate the maximum norm of the inverse of A
    cond = norm_A * norm_A_inv  # Compute the condition number of matrix A
    
    Return cond  # Return the condition number of matrix A

"""


def norm(mat):
    size = len(mat)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(mat[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row


def condition_number(A):
    # Step 1: Calculate the max norm (infinity norm) of A
    norm_A = norm(A)

    # Step 2: Calculate the inverse of A
    A_inv = inverse(A)

    # Step 3: Calculate the max norm of the inverse of A
    norm_A_inv = norm(A_inv)

    # Step 4: Compute the condition number
    cond = norm_A * norm_A_inv

    # print the values
    """
    print(bcolors.OKBLUE, "A:", bcolors.ENDC)
    print_matrix(A)

    print(bcolors.OKBLUE, "inverse of A:", bcolors.ENDC)
    print_matrix(A_inv)

    print(bcolors.OKBLUE, "Max Norm of A:", bcolors.ENDC, norm_A, "\n")

    print(bcolors.OKBLUE, "max norm of the inverse of A:", bcolors.ENDC, norm_A_inv)"""
    return cond
if __name__ == '__main__':
    A = np.array([[2, 1.7, -2.5],
                  [1.24, -2, -0.5],
                  [3, 0.2, 1]])
    cond = condition_number(A)

    print(bcolors.OKGREEN, "\n condition number: ", cond, bcolors.ENDC)
