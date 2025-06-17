import abc
from dolfin import *


class ProblemDefinition(abc.ABC):
    """
    Base class for a generic PDE problem in its weak formulation.
    """
    @abc.abstractmethod
    def a(self, u, v):
        pass

    @abc.abstractmethod
    def L(self, v):
        pass

    @abc.abstractmethod
    def g(self):
        pass


class PoissonProblem(ProblemDefinition):

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def a(self, u, v):
        return inner(grad(u), grad(v))*dx

    def L(self, v):
        return self.f * v * dx

    def g(self):
        return self.g


class HelmholtzProblem(ProblemDefinition):

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def a(self, u, v):
        return (inner(grad(u), grad(v)) + u*v) * dx

    def L(self, v):
        return self.f * v * dx

    def g(self):
        return self.g

class LinearElasticity(ProblemDefinition):

    def __init__(self, g):
        self.g = g
        self.E = 10**7
        self.nu = 0.3
        self.A = -self.E / (2*(1+self.nu))
        self.B = self.A / (1-2*self.nu)
        self.C = - (30*self.E*(1-self.nu)) / ((1+self.nu)*(1-2*self.nu))

    def a(self, u, v):
        return (-self.A * inner(grad(u), grad(v)) - self.B * div(u) * div(v)) * dx

    def L(self, v):
        f = Expression(("C*pow(x[0], 4)", "C*pow(x[1], 4)"), C=self.C, degree=2)
        return inner(f, v) * dx

    def g(self):
        return self.g


