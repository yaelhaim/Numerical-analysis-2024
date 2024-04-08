import jacobi_utilities
from sympy import *
from matrix_utility import *
from sympy import symbols, lambdify

import numpy as np


x = Symbol('x')

def natural_cubic_spline(f, x0):
    print("\033[94m" + "Natural Cubic Spline" + "\033[0m")
    x = Symbol('x')
    h = list()
    for i in range(len(f) - 1):
        h.append(f[i + 1][0] - f[i][0])
    g = list()
    g.append(0)  # g0
    for i in range(1, len(f) - 1):
        g.append(h[i] / (h[i] + h[i - 1]))
    g.append(0)  # gn
    m = list()
    m.append(0)
    for i in range(1, len(f)):
        m.append(1 - g[i])
    d = list()
    d.append(0)  # d0=0
    for i in range(1, len(f) - 1):
        d.append((6 / (h[i - 1] + h[i])) * (((f[i + 1][1] - f[i][1]) / h[i]) - ((f[i][1] - f[i - 1][1]) / h[i - 1])))
    d.append(0)  # dn
    # building the matrix
    mat = list()
    # first row
    mat.append(list())
    mat[0].append(2)
    for j in range(len(f) - 1):
        mat[0].append(0)
    for i in range(1, len(f) - 1):
        mat.append(list())
        for j in range(len(f)):
            if j == i - 1:  # put miu
                mat[i].append(m[i])
            elif j == i:
                mat[i].append(2)
            elif j == i + 1:  # put lambda
                mat[i].append(g[i])
            else:
                mat[i].append(0)
    # last row
    mat.append(list())
    for j in range(len(f) - 1):
        mat[len(f) - 1].append(0)
    mat[len(f) - 1].append(2)
    print("matrix: " + str(mat))
    print("vector b: " + str(d))
    # get m vector
    print("\nJacobi middle results: ")
    M = (jacobi_utilities.Jacobi(mat, d))
    print("\nvector M: " + str(list(map(float, M))))
    # find S:
    for loc in range(1, len(f)):
        s = (((f[loc][0] - x) ** 3) * M[loc - 1] + ((x - f[loc - 1][0]) ** 3) * M[loc]) / (6 * h[loc - 1])
        s += (((f[loc][0] - x) * f[loc - 1][1]) + ((x - f[loc - 1][0]) * f[loc][1])) / h[loc - 1]
        s -= (((f[loc][0] - x) * M[loc - 1] + (x - f[loc - 1][0]) * M[loc]) * h[loc - 1]) / 6
        print("s" + str(loc - 1) + "(x) = " + str(s))
    # find the location of x0:
    loc = 0
    for i in range(1, len(f)):
        if x0 <= f[i][0] and x0 >= f[i - 1][0]:
            loc = i
            break
    if loc == 0:
        print("no range found for x0")
        return
    s = (((f[loc][0] - x) ** 3) * M[loc - 1] + ((x - f[loc - 1][0]) ** 3) * M[loc]) / (6 * h[loc - 1])
    s += (((f[loc][0] - x) * f[loc - 1][1]) + ((x - f[loc - 1][0]) * f[loc][1])) / h[loc - 1]
    s -= (((f[loc][0] - x) * M[loc - 1] + (x - f[loc - 1][0]) * M[loc]) * h[loc - 1]) / 6
    print("\nx0 between f(x" + str(loc - 1) + ") = " + str(f[loc - 1][0]) + " and f(x" + str(loc) + ") = " + str(
        f[loc][0]) + " so:")
    print("s" + str(loc - 1) + "(" + str(x0) + ") = " + str(float(s.subs(x, x0))))
    s = lambdify(x, s)
    print("The sol")
    print(s(x0))


def CubicSplineFtag(tableValue, X, Ftag = None):
    size = len(tableValue)
    matrix = np.zeros((size, size))
    b = np.zeros(size)
    h = [tableValue[i+1][0] - tableValue[i][0] for i in range(size - 1)]
    if Ftag == None:
        matrix[0][0] = 1
        b[0] = 0
        matrix[-1][-1] = 1
        b[-1] = 0
    else:
        matrix[0][0] = 1/3 * h[0]
        matrix[0][1] = 1/6 * h[0]
        b[0] = ((tableValue[1][1] - tableValue[0][1])/h[0]) - Ftag[0]
        matrix[-1][-1] = 1/3 * h[-1]
        matrix[-1][-2] = 1/6 * h[-1]
        b[-1] = Ftag[1] - ((tableValue[-1][1] - tableValue[-2][1]) / h[-1])
    for i in range(1, size - 1):
        for j in range(i - 1, min(i + 2, size)):
            if i > j:
                matrix[i][j] = 1/6 * h[j]
            else:
                if i < j:
                    matrix[i][j] = 1 / 6 * h[j-1]
                else:
                    matrix[i][j] = 1/3 * (h[i-1] + h[i])
        b[i] = ((tableValue[i+1][1] - tableValue[i][1]) / h[i]) - ((tableValue[i][1] - tableValue[i-1][1])/h[i-1])
    matrixNew = np.hstack((matrix, b.reshape(-1, 1)))
    matrixSol = gaussianElimination(matrixNew)
    x = Symbol('x')
    for i in range(1, size):
        sum = (((tableValue[i][0] - x) ** 3) * matrixSol[i - 1] + ((x - tableValue[i - 1][0]) ** 3) * matrixSol[i]) / (6 * h[i - 1])
        sum += (((tableValue[i][0] - x) * tableValue[i - 1][1]) + ((x - tableValue[i - 1][0]) * tableValue[i][1])) / h[i - 1]
        sum -= (((tableValue[i][0] - x) * matrixSol[i - 1] + (x - tableValue[i - 1][0]) * matrixSol[i]) * h[i - 1]) / 6
        print("s" + str(i - 1) + "(x) = " + str(sum))
    loc = 0
    for i in range(1, len(f)):
        if X <= tableValue[i][0] and X >= tableValue[i - 1][0]:
            loc = i
            break
    if loc == 0:
        print("no range found for x0")
        return
    sol = (((tableValue[loc][0] - x) ** 3) * matrixSol[loc - 1] + ((x - tableValue[loc - 1][0]) ** 3) * matrixSol[loc]) / (6 * h[loc - 1])
    sol += (((tableValue[loc][0] - x) * tableValue[loc - 1][1]) + ((x - tableValue[loc - 1][0]) * tableValue[loc][1])) / h[loc - 1]
    sol -= (((tableValue[loc][0] - x) * matrixSol[loc - 1] + (x - tableValue[loc - 1][0]) * matrixSol[loc]) * h[loc - 1]) / 6
    print(f"\nx0 = {X} between f(x" + str(loc - 1) + ") = " + str(tableValue[loc - 1][0]) + " and f(x" + str(loc) + ") = " + str(tableValue[loc][0]) + " so:")
    print("s" + str(loc - 1) + "(" + str(X) + ") = " + "\033[94m" +str(float(sol.subs(x, X))) + "\033[0m")

if __name__ == '__main__':
    fE = [(-1,1), (0,0), (1,1)]
    x0 = 0.5
    Ftag = (-3, 3)  # [S'(a), S'(b)]
    """natural_cubic_spline(fE, x0)"""
    CubicSplineFtag(fE,x0,Ftag)
    x = [-1, 0, 1]
    y = [1, 0, 1]


