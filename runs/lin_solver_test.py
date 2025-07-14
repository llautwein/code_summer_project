import numpy as np
import LinearSystemSolver as lss

A = np.array([[7, 6],
              [1, 10]])
b = np.array([1, 2])

x_exact = np.linalg.solve(A, b)
print(x_exact)
jacobi = lss.JacobiMethod(A, b)
x_0 = np.array([1, 1])
x_J = jacobi.solve(x_0, 1e-5, 100)
print(x_J)

gauss_seidel = lss.GaussSeidelMethod(A, b)
x_GS = gauss_seidel.solve(x_0, 1e-5, 100)
print(x_GS)

gmres = lss.GMRESMethod(A, b, 5)
x_GM = gmres.solve(x_0, 1e-5, 100)
print(x_GM)
