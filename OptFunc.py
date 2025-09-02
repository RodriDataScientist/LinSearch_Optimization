import numpy as np
import sympy as sp

class OptFunc:
    """
    Wrapper simbólico/numérico para función, gradiente y Hessiano.
    Puede recibir:
      - func_expr: sympy.Expr  o  callable(x: array)->float
      - grad_expr: list[Expr]  o  callable(x: array)->array
      - hess_expr: Matrix/2D list  o  callable(x: array)->(n,n)
    Si no se pasan grad/hess, se intentan construir (solo si func_expr es simbólico).
    """
    def __init__(self, func_expr=None, vars=None, grad_expr=None, hess_expr=None):
        if vars is None:
            raise ValueError("Debe proporcionar 'vars' (puede ser lista de símbolos o índices).")
        self.vars = tuple(vars)
        self.n = len(self.vars)

        self.func_expr = func_expr
        self.grad_expr = grad_expr
        self.hess_expr = hess_expr

        # ---- función ----
        if callable(func_expr):
            self._func = func_expr
        elif isinstance(func_expr, sp.Expr):
            self._func = sp.lambdify(self.vars, func_expr, "numpy")
        elif func_expr is None:
            self._func = None
        else:
            raise TypeError("func_expr debe ser callable, sympy.Expr o None.")

        # ---- gradiente ----
        if grad_expr is None:
            if isinstance(func_expr, sp.Expr):
                grad_expr = [sp.diff(func_expr, v) for v in self.vars]
            else:
                grad_expr = None  # no disponible si función no es simbólica

        if callable(grad_expr):
            self._grad = grad_expr
        elif grad_expr is None:
            self._grad = None
        else:
            # lista/tupla de sympy.Expr
            self._grad = sp.lambdify(self.vars, list(grad_expr), "numpy")

        # ---- Hessiano ----
        if hess_expr is None:
            if isinstance(func_expr, sp.Expr):
                hess_expr = sp.hessian(func_expr, self.vars)
            elif isinstance(grad_expr, (list, tuple)) and all(isinstance(g, sp.Expr) for g in grad_expr):
                hess_expr = sp.Matrix([[sp.diff(g, v) for v in self.vars] for g in grad_expr])
            else:
                hess_expr = None

        if callable(hess_expr):
            self._hess = hess_expr
        elif hess_expr is None:
            self._hess = None
        else:
            if isinstance(hess_expr, (list, tuple)):
                hess_expr = sp.Matrix(hess_expr)
            if not isinstance(hess_expr, sp.MatrixBase):
                raise TypeError("hess_expr debe ser callable, Matrix o lista de listas.")
            self._hess = sp.lambdify(self.vars, hess_expr, "numpy")

        self.domain = None
        self.closed = (True, True)

    # -----------------------
    # Evaluadores numéricos
    # -----------------------
    def _check_x(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n:
            raise ValueError(f"Entrada de dimensión {x.size}, se esperaba {self.n}")
        return x

    def eval(self, x):
        x = self._check_x(x)
        if self._func is None:
            raise RuntimeError("Función no disponible.")
        val = self._func(x) if self._expects_vector(self._func) else self._func(*x)
        return float(np.asarray(val).ravel()[0])

    def gradient(self, x):
        x = self._check_x(x)
        if self._grad is None:
            raise RuntimeError("Gradiente no disponible.")
        g = self._grad(x) if self._expects_vector(self._grad) else self._grad(*x)
        g = np.asarray(g, dtype=float).ravel()
        if g.size != self.n:
            raise RuntimeError(f"Gradiente devuelto de tamaño {g.size}, esperado {self.n}")
        return g

    def hessian(self, x):
        x = self._check_x(x)
        if self._hess is None:
            raise RuntimeError("Hessiano no disponible.")
        H = self._hess(x) if self._expects_vector(self._hess) else self._hess(*x)
        H = np.asarray(H, dtype=float)
        if H.shape != (self.n, self.n):
            # a veces puede venir aplanado
            if H.size == self.n*self.n:
                H = H.reshape((self.n, self.n))
            else:
                raise RuntimeError(f"Hessiano con forma {H.shape}, esperado {(self.n, self.n)}")
        return H

    @staticmethod
    def _expects_vector(fn):
        # Heurística: si la función tiene un único argumento, asumimos vector
        try:
            import inspect
            sig = inspect.signature(fn)
            return len(sig.parameters) == 1
        except Exception:
            return False