import warnings
import numpy as np
import matplotlib.pyplot as plt

from OptFunc import OptFunc

class LineSearch:
    def __init__(self, function: OptFunc,
                 stop_crit=None,
                 step_cond="goldstein",   
                 desc_cond="gradient_descent",
                 x0=None,
                 alpha=1.0,
                 max_iter=1000,
                 tol=1e-6,
                 rho=0.5,
                 c1=1e-4,
                 c2=0.8):
        self.function = function
        self.step_cond = step_cond
        self.desc_cond = desc_cond
        self.stop_crit = stop_crit if stop_crit is not None else {}
        self.x0 = np.asarray(x0, dtype=float) if x0 is not None else np.zeros(self.function.n)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rho = float(rho)
        self.c1 = float(c1)
        self.c2 = float(c2)

        # Estado para BFGS (si se usa)
        self._B_inv = None
        self._prev_x = None
        self._prev_grad = None
        if (not callable(self.desc_cond)) and (self.desc_cond == "bfgs"):
            self._B_inv = np.eye(self.function.n)

        # Mapas para evitar if/elif repetidos
        self._step_map = {
            "backtracking": self._backtracking,
            "wolfe": self._wolfe,
            "goldstein": self._goldstein,
            "fixed": lambda x, p: self.alpha,
            None: lambda x, p: self.alpha
        }
        self._desc_map = {
            "gradient_descent": self._desc_gd,
            "newton": self._desc_newton,
            "bfgs": self._desc_bfgs
        }

    # Line search invocable
    def __call__(self, x, p):
        # permite paso personalizado por callable
        if callable(self.step_cond):
            return self.step_cond(x, p)
        # usa mapa por nombre
        fn = self._step_map.get(self.step_cond, None)
        if fn is None:
            raise ValueError(f"step_cond '{self.step_cond}' no reconocido")
        return fn(x, p)

    # Stop criterion
    def _stop(self, k, grad):
        if callable(self.stop_crit):
            return self.stop_crit(k, grad)
        tol = self.stop_crit.get('tol', self.tol)
        max_iter = self.stop_crit.get('max_iter', self.max_iter)
        if np.linalg.norm(grad) <= tol:
            return True
        if k >= max_iter:
            return True
        return False

    # Direcciones de descenso
    def _direction(self, x, grad):
        # permite callable externo para desc_cond
        if callable(self.desc_cond):
            return self.desc_cond(x, grad)
        fn = self._desc_map.get(self.desc_cond, None)
        if fn is None:
            raise ValueError(f"Método de descenso '{self.desc_cond}' no reconocido")
        return fn(x, grad)

    def _desc_gd(self, x, grad):
        return -grad

    def _desc_newton(self, x, grad):
        H = self.function.hessian(x)
        try:
            return -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            H_reg = H + 1e-8 * np.eye(H.shape[0])
            warnings.warn("Hessian singular: applied tiny damping to Hessian (newton).")
            return -np.linalg.solve(H_reg, grad)

    def _desc_bfgs(self, x, grad):
        # Si _B_inv no inicializado (p. ej. se pasó callable), fallback a -grad
        if self._B_inv is None:
            return -grad
        return -self._B_inv @ grad

    # Backtracking / Wolfe / Goldstein 
    def _phi(self, x, p, alpha):
        return self.function.eval(x + alpha * p)

    def _phi_prime(self, x, p, alpha):
        return np.dot(self.function.gradient(x + alpha * p), p)

    def _backtracking(self, x, p, alpha0=None, c=None):
        alpha = self.alpha if alpha0 is None else alpha0
        c = self.c1 if c is None else c
        f0 = self.function.eval(x)
        g0 = np.dot(self.function.gradient(x), p)
        while True:
            fnew = self.function.eval(x + alpha * p)
            if fnew <= f0 + c * alpha * g0:
                break
            alpha *= self.rho
            if alpha < 1e-12:
                break
        return alpha

    def _zoom_wolfe(self, x, p, alo, ahi, phi0, phi0_prime, c1, c2, mode="strong", maxiter=50):
        for _ in range(maxiter):
            aj = 0.5 * (alo + ahi)
            phi_aj = self._phi(x, p, aj)
            phi_alo = self._phi(x, p, alo)
            if (phi_aj > phi0 + c1 * aj * phi0_prime) or (phi_aj >= phi_alo):
                ahi = aj
            else:
                phi_prime_aj = self._phi_prime(x, p, aj)
                if (mode == "strong" and abs(phi_prime_aj) <= c2 * abs(phi0_prime)) or \
                   (mode != "strong" and phi_prime_aj >= c2 * phi0_prime):
                    return aj
                if phi_prime_aj * (ahi - alo) >= 0:
                    ahi = alo
                alo = aj
        return 0.5 * (alo + ahi)

    def _line_search_wolfe(self, x, p, alpha0=None, c1=None, c2=None, mode="armijo", maxiter=50):
        alpha0 = self.alpha if alpha0 is None else alpha0
        c1 = self.c1 if c1 is None else c1
        c2 = self.c2 if c2 is None else c2
        phi0 = self._phi(x, p, 0.0)
        phi0_prime = self._phi_prime(x, p, 0.0)
        if mode == "armijo":
            return self._backtracking(x, p, alpha0, c1)
        alpha_prev = 0.0
        alpha = alpha0
        phi_prev = phi0
        for i in range(maxiter):
            phi_alpha = self._phi(x, p, alpha)
            if (phi_alpha > phi0 + c1 * alpha * phi0_prime) or (i > 0 and phi_alpha >= phi_prev):
                return self._zoom_wolfe(x, p, alpha_prev, alpha, phi0, phi0_prime, c1, c2, mode=mode)
            phi_prime_alpha = self._phi_prime(x, p, alpha)
            if (mode == "weak" and phi_prime_alpha >= c2 * phi0_prime) or \
               (mode != "weak" and abs(phi_prime_alpha) <= c2 * abs(phi0_prime)):
                return alpha
            if phi_prime_alpha >= 0:
                return self._zoom_wolfe(x, p, alpha, alpha_prev, phi0, phi0_prime, c1, c2, mode=mode)
            alpha_prev = alpha
            phi_prev = phi_alpha
            alpha = alpha * 2.0
            if alpha > 1e8:
                break
        return self._backtracking(x, p, alpha0, c1)

    def _line_search_goldstein(self, x, p, alpha0=None, c=0.1, maxiter=50):
        alpha0 = self.alpha if alpha0 is None else alpha0
        c = 0.1 if c is None else c
        phi0 = self._phi(x, p, 0.0)
        g0_dot_p = self._phi_prime(x, p, 0.0)
        if g0_dot_p >= 0:
            warnings.warn("p no es dirección de descenso en Goldstein; devolviendo alpha por defecto.")
            return self.alpha
        alpha_lo = 0.0
        alpha = alpha0
        for _ in range(maxiter):
            phi_alpha = self._phi(x, p, alpha)
            lower = phi0 + (1 - c) * alpha * g0_dot_p
            upper = phi0 + c * alpha * g0_dot_p
            if lower <= phi_alpha <= upper:
                return alpha
            if phi_alpha < lower:
                return self._zoom_goldstein(x, p, alpha_lo, alpha, phi0, g0_dot_p, c)
            alpha_lo = alpha
            alpha *= 2.0
        return self.alpha

    def _zoom_goldstein(self, x, p, alo, ahi, phi0, g0_dot_p, c, maxiter=50):
        for _ in range(maxiter):
            aj = 0.5 * (alo + ahi)
            phi_aj = self._phi(x, p, aj)
            lower = phi0 + (1 - c) * aj * g0_dot_p
            upper = phi0 + c * aj * g0_dot_p
            if lower <= phi_aj <= upper:
                return aj
            if phi_aj < lower:
                ahi = aj
            else:
                alo = aj
        return 0.5 * (alo + ahi)

    def _wolfe(self, x, p):
        return self._line_search_wolfe(x, p)
    def _goldstein(self, x, p):
        return self._line_search_goldstein(x, p)

    def _bfgs_update(self, s, y):
        ys = float(y.T @ s)
        if ys > 1e-12:
            rho = 1.0 / ys
            I = np.eye(self._B_inv.shape[0])
            self._B_inv = (I - rho * s @ y.T) @ self._B_inv @ (I - rho * y @ s.T) + rho * s @ s.T
        else:
            warnings.warn("salteada actualización BFGS por ys muy pequeño.")

    def optimize(self, return_history=False, return_trajectory=False):
        x = np.asarray(self.x0, dtype=float).ravel()
        n = x.size
        history = []
        trajectory = []

        for k in range(int(self.max_iter)):
            grad = self.function.gradient(x)
            fval = self.function.eval(x)
            history.append({'k': k, 'x': x.copy(), 'f': fval, 'grad_norm': np.linalg.norm(grad)})
            trajectory.append(x.copy())

            if self._stop(k, grad):
                break

            # dirección (soporta callable o string)
            p = self._direction(x, grad)
            if np.dot(grad, p) >= 0:
                warnings.warn("dirección no es de descenso; usando -grad como fallback.")
                p = -grad

            # búsqueda de paso (invoca __call__)
            alpha = self(x, p)
            if alpha is None or alpha <= 0:
                alpha = max(self.alpha * 1e-6, 1e-12)

            x_new = x + alpha * p
            grad_new = self.function.gradient(x_new)

            # actualiza BFGS si corresponde
            if (not callable(self.desc_cond)) and (self.desc_cond == "bfgs"):
                s = (x_new - x).reshape(-1, 1)
                y = (grad_new - grad).reshape(-1, 1)
                self._bfgs_update(s, y)

            # guarda prev para la clase si el usuario pasó callable bfgs externa
            self._prev_x = x.copy()
            self._prev_grad = grad.copy()

            x = x_new

        trajectory.append(x.copy())

        if return_history and return_trajectory:
            return x, history, np.array(trajectory)
        if return_history:
            return x, history
        if return_trajectory:
            return x, np.array(trajectory)
        return x
    
    def plot_trajectory(self,
                        history=None,
                        trajectory=None,
                        proj_method='pca',
                        components=(0, 1),
                        grid_points=60,
                        padding=0.2,
                        show_contours=True,
                        levels=50,
                        figsize=(8, 6)):
        # obtener trayectoria
        if trajectory is None:
            if history is None:
                raise ValueError("Proporciona 'history' o 'trajectory' para plotear")
            traj = np.array([h['x'] for h in history])
        else:
            traj = np.asarray(trajectory)

        if traj.ndim != 2:
            raise ValueError("trajectory debe ser array 2D (steps, n)")

        steps, n = traj.shape

        if proj_method == 'coords' and n < 2:
            raise ValueError("No hay suficientes dimensiones para projection 'coords'.")

        if proj_method == 'coords':
            i, j = components
            proj = traj[:, [i, j]]

            # grid en las dos coordenadas seleccionadas
            mins = proj.min(axis=0)
            maxs = proj.max(axis=0)
            ranges = maxs - mins
            mins = mins - padding * ranges
            maxs = maxs + padding * ranges

            u = np.linspace(mins[0], maxs[0], grid_points)
            v = np.linspace(mins[1], maxs[1], grid_points)
            U, V = np.meshgrid(u, v)

            Z = None
            if show_contours:
                Z = np.empty_like(U)
                # para cada punto en la grilla construimos un x en el espacio original
                for ii in range(grid_points):
                    for jj in range(grid_points):
                        x_map = traj.mean(axis=0).copy()
                        x_map[i] = U[ii, jj]
                        x_map[j] = V[ii, jj]
                        Z[ii, jj] = self.function.eval(x_map)

        else:  # proj_method == 'pca'
            mean = traj.mean(axis=0)
            U_svd, S, Vt = np.linalg.svd(traj - mean, full_matrices=False)
            comps = Vt.T[:, :2]  # n x 2
            proj = (traj - mean) @ comps

            mins = proj.min(axis=0)
            maxs = proj.max(axis=0)
            ranges = maxs - mins
            mins = mins - padding * ranges
            maxs = maxs + padding * ranges

            u = np.linspace(mins[0], maxs[0], grid_points)
            v = np.linspace(mins[1], maxs[1], grid_points)
            U, V = np.meshgrid(u, v)

            Z = None
            if show_contours:
                Z = np.empty_like(U)
                # mapear cada punto 2D de vuelta al espacio original: x = mean + comps @ [u,v]
                for ii in range(grid_points):
                    for jj in range(grid_points):
                        uv = np.array([U[ii, jj], V[ii, jj]])
                        x_map = mean + comps @ uv
                        Z[ii, jj] = self.function.eval(x_map)

        fig, ax = plt.subplots(figsize=figsize)

        if show_contours and Z is not None:
            cs = ax.contour(U, V, Z, levels=levels, cmap='viridis')
            ax.clabel(cs, inline=True, fontsize=8) 

        ax.plot(proj[:, 0], proj[:, 1], '-o', markersize=4, label='trajectory')
        ax.scatter(proj[0, 0], proj[0, 1], s=80, marker='D', label='start')
        ax.scatter(proj[-1, 0], proj[-1, 1], s=80, marker='X', label='end')
        ax.set_xlabel('PC1' if proj_method == 'pca' else f'x{components[0]}')
        ax.set_ylabel('PC2' if proj_method == 'pca' else f'x{components[1]}')
        
        n_iter = traj.shape[0] - 1
        stop_info = self.stop_crit if self.stop_crit else f"tol={self.tol}, max_iter={self.max_iter}"
        title = (f"Trajectory (projected)\n"
                 f"descenso={self.desc_cond}, paso={self.step_cond}, "
                 f"stop={stop_info}, iters={n_iter}")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        return fig, ax, proj