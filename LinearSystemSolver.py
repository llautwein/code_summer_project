import numpy as np


class LinearSystemSolver:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.iterates = []
        self.errors = []
        self.n = A.shape[0]
        np.set_printoptions(precision=5)

    def print_log(self, k):
        print(f"Iteration {k}: x_k = {self.iterates[k]}, err = {round(self.errors[k-1], 9)}")

    def solve(self, x_0, tol, max_iter):
        pass

class JacobiMethod(LinearSystemSolver):
    def __init__(self, A, b):
        super().__init__(A, b)

    def solve(self, x_0, tol, max_iter):
        self.iterates.append(x_0)
        self.errors.append(1)
        for k in range(max_iter):
            self.print_log(k)
            if self.errors[k] < tol and k > 0:
                print("Tolerance reached!")
                return self.iterates, self.errors
            x_k = np.zeros(self.n)
            for i in range(self.n):
                sum = 0
                for j in range(self.n):
                    if j == i:
                        continue
                    sum -= self.A[i, j] * self.iterates[k][j]
                x_k[i] += (sum + self.b[i]) / self.A[i, i]
            self.iterates.append(x_k)
            self.errors.append(np.linalg.norm(self.iterates[k+1]-self.iterates[k]))
        return self.iterates, self.errors

class GaussSeidelMethod(LinearSystemSolver):
    def __init__(self, A, b):
        super().__init__(A, b)

    def solve(self, x_0, tol, max_iter):
        self.iterates.append(x_0)
        self.errors.append(1)
        for k in range(max_iter):
            self.print_log(k)
            if self.errors[k] < tol and k > 0:
                print("Tolerance reached!")
                return self.iterates, self.errors
            x_k = np.zeros(self.n)
            for i in range(self.n):
                sum1 = 0
                sum2 = 0
                for j in range(self.n):
                    if j < i:
                        sum1 -= self.A[i, j] * x_k[j]
                    if j > i:
                        sum2 -= self.A[i, j] * self.iterates[k][j]
                x_k[i] += (sum1 + sum2 + self.b[i]) / self.A[i, i]
            self.iterates.append(x_k)
            self.errors.append(np.linalg.norm(self.iterates[k + 1] - self.iterates[k]))
        return self.iterates, self.errors

class GMRESMethod(LinearSystemSolver):
    def __init__(self, A, b, m):
        super().__init__(A, b)
        self.m = m

    def arnoldi_iteration(self, r_k):
        Q = np.zeros((self.n, self.m+1))
        m_eff = self.m
        Q[:, 0] = r_k / np.linalg.norm(r_k)
        H = np.zeros((self.m+1, self.m))
        for j in range(self.m):
            v = self.A @ Q[:, j]
            for i in range(j):
                H[i, j] = np.dot(Q[:, i], v)
                v -= H[i, j] * Q[:, i]
            H[j+1, j] = np.linalg.norm(v)
            if H[j + 1, j] < 1e-12:
                m_eff = j + 1  # Subspace dimension is smaller than m
                break
            Q[:, j+1] = v / H[j+1, j]
        return Q[:, :m_eff+1], H[:m_eff+1, :m_eff], m_eff

    def solve(self, x_0, tol, max_iter):
        self.iterates.append(x_0)
        x_k = x_0.astype(float).copy()
        r_k = self.b - self.A @ x_0
        self.errors.append(np.linalg.norm(r_k))
        for k in range(max_iter):
            self.print_log(k)
            if self.errors[k] < tol:
                print("Tolerance reached!")
                return self.iterates, self.errors
            beta = self.errors[k]
            Q, H, m_eff = self.arnoldi_iteration(r_k)
            e1 = np.zeros(m_eff + 1)
            e1[0] = 1
            g = beta * e1
            y_sol, _, _, _ = np.linalg.lstsq(H, g, rcond=None)
            x_k += Q[:, :m_eff] @ y_sol
            self.iterates.append(x_k.copy())
            r_k = self.b - self.A @ x_k
            self.errors.append(np.linalg.norm(r_k))
        return self.iterates, self.errors





