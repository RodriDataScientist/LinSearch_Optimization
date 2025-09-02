import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from LinSearch import LineSearch
from OptFunc import OptFunc

def get_test_functions(d=10):
    funcs = {}

    # Símbolos
    x = sp.symbols(f'x0:{d}')

    # 1) Rosenbrock d-dim
    rosen_expr = sum((1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(d-1))
    rosen_grad = [sp.diff(rosen_expr, xi) for xi in x]
    rosen_hess = sp.hessian(rosen_expr, x)
    rosen = OptFunc(rosen_expr, x, grad_expr=rosen_grad, hess_expr=rosen_hess)
    funcs["rosenbrock"] = (rosen, np.ones(d))

    # 2) Ackley d-dim
    sqr_sum = sum(xi**2 for xi in x)
    cos_sum = sum(sp.cos(2*sp.pi*xi) for xi in x)
    ackley_expr = -20*sp.exp(-0.2*sp.sqrt(sqr_sum/d)) - sp.exp(cos_sum/d) + 20 + sp.E
    ackley_grad = [sp.diff(ackley_expr, xi) for xi in x]
    ackley_hess = sp.hessian(ackley_expr, x)
    ackley = OptFunc(ackley_expr, x, grad_expr=ackley_grad, hess_expr=ackley_hess)
    funcs["ackley"] = (ackley, np.zeros(d))

    # 3) Griewank d-dim
    sum_term = sum(xi**2 / 4000 for xi in x)
    prod_term = 1
    # build product term symbolically
    prod_term = sp.prod([sp.cos(x[i]/sp.sqrt(i+1)) for i in range(d)])
    griewank_expr = sum_term - prod_term + 1
    griewank_grad = [sp.diff(griewank_expr, xi) for xi in x]
    griewank_hess = sp.hessian(griewank_expr, x)
    griewank = OptFunc(griewank_expr, x, grad_expr=griewank_grad, hess_expr=griewank_hess)
    funcs["griewank"] = (griewank, np.zeros(d))

    return funcs

def get_test_functions_manual(d=10):
    funcs = {}

    # ---------------------------
    # 1) ROSENBROCK (manual)
    # f = sum_{i=0}^{d-2} (1 - x_i)^2 + 100*(x_{i+1} - x_i^2)^2
    # grad:
    #  i = 0:      g0 = -2(1-x0) - 400 x0 (x1 - x0^2)
    #  1<=i<=d-2:  gi = -2(1-xi) - 400 xi (x_{i+1}-xi^2) + 200(xi - x_{i-1}^2)
    #  i = d-1:    g_{d-1} = 200(x_{d-1} - x_{d-2}^2)
    #
    # Hessiano tridiagonal:
    #  H[i,i]:
    #   i=0:         1200 x0^2 - 400 x1 + 2
    #   1<=i<=d-2:   1200 xi^2 - 400 x_{i+1} + 202
    #   i=d-1:       200
    #  H[i,i+1]=H[i+1,i]= -400 x_i
    #
    def rosen_f(x):
        x = np.asarray(x, dtype=float)
        s = np.sum((1 - x[:-1])**2 + 100.0*(x[1:] - x[:-1]**2)**2)
        return float(s)

    def rosen_g(x):
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        # i = 0
        g[0] = -2.0*(1.0 - x[0]) - 400.0*x[0]*(x[1] - x[0]**2)
        # 1 .. d-2
        if x.size > 2:
            i = np.arange(1, x.size-1)
            g[i] = (-2.0*(1.0 - x[i])
                    - 400.0*x[i]*(x[i+1] - x[i]**2)
                    + 200.0*(x[i] - x[i-1]**2))
        # i = d-1
        g[-1] = 200.0*(x[-1] - x[-2]**2)
        return g

    def rosen_H(x):
        x = np.asarray(x, dtype=float)
        n = x.size
        H = np.zeros((n, n), dtype=float)

        # diagonales
        if n >= 1:
            H[0, 0] = 1200.0*x[0]**2 - 400.0*x[1] + 2.0
        if n >= 2:
            H[-1, -1] = 200.0
        if n > 2:
            i = np.arange(1, n-1)
            H[i, i] = 1200.0*x[i]**2 - 400.0*x[i+1] + 202.0

        # fuera de diagonal (tridiagonal)
        if n >= 2:
            i = np.arange(0, n-1)
            H[i, i+1] = -400.0*x[i]
            H[i+1, i] = -400.0*x[i]
        return H

    rosen = OptFunc(func_expr=rosen_f, vars=list(range(d)),
                    grad_expr=rosen_g, hess_expr=rosen_H)
    funcs["rosenbrock"] = (rosen, np.ones(d))

    # ---------------------------
    # 2) ACKLEY (manual)
    # f = -20 exp(-0.2*sqrt(S/d)) - exp(C/d) + 20 + e
    #   S = sum xi^2
    #   C = sum cos(2π xi)
    #
    # grad_i = 4*exp(-0.2 r) * xi / (d*r)  +  (2π/d) * exp(C/d) * sin(2π xi)
    #   donde r = sqrt(S/d). Usar eps para r=0.
    #
    # Hessiano:
    #  Sea E1 = exp(-0.2 r), E2 = exp(C/d)
    #  A_ij (del término de S):
    #    H_A[i,j] = (4/d) * E1 * [ δ_ij / r  - (xi*xj)/(d*r**3)  - 0.2*(xi*xj)/(d*r**2) ]
    #  B_ij (del término de C):
    #    H_B[i,j] = E2 * [ -(4π^2)/(d^2) sin(2π xj) sin(2π xi)  +  (4π^2)/d * δ_ij cos(2π xi) ]
    #
    def ackley_f(x):
        x = np.asarray(x, dtype=float)
        d_ = x.size
        S = np.dot(x, x)
        C = np.sum(np.cos(2.0*np.pi*x))
        term1 = -20.0 * np.exp(-0.2*np.sqrt(S / d_))
        term2 = -np.exp(C / d_)
        return float(term1 + term2 + 20.0 + np.e)

    def ackley_g(x):
        x = np.asarray(x, dtype=float)
        d_ = x.size
        S = np.dot(x, x)
        r = np.sqrt(S / d_)
        eps = 1e-12
        r_safe = max(r, eps)

        E1 = np.exp(-0.2*r_safe)
        E2 = np.exp(np.sum(np.cos(2.0*np.pi*x)) / d_)

        termA = 4.0 * E1 * x / (d_ * r_safe)
        termB = (2.0*np.pi / d_) * E2 * np.sin(2.0*np.pi*x)
        return termA + termB

    def ackley_H(x):
        x = np.asarray(x, dtype=float)
        d_ = x.size
        S = np.dot(x, x)
        r = np.sqrt(S / d_)
        eps = 1e-12
        r_safe = max(r, eps)

        E1 = np.exp(-0.2*r_safe)
        E2 = np.exp(np.sum(np.cos(2.0*np.pi*x)) / d_)

        # Matrices auxiliares
        H = np.zeros((d_, d_), dtype=float)
        # A: parte por S
        # diag delta_ij / r
        H += (4.0/d_) * E1 * (np.eye(d_) / r_safe)

        # - (xi*xj)/(d*r^3)  - 0.2*(xi*xj)/(d*r^2)
        outer_xx = np.outer(x, x)
        H -= (4.0/d_) * E1 * (outer_xx / (d_ * (r_safe**3)))
        H -= (4.0/d_) * E1 * (0.2 * outer_xx / (d_ * (r_safe**2)))

        # B: parte por C
        sin2p = np.sin(2.0*np.pi*x)
        cos2p = np.cos(2.0*np.pi*x)
        H -= E2 * (4.0*(np.pi**2)/(d_**2)) * np.outer(sin2p, sin2p)
        H += E2 * (4.0*(np.pi**2)/d_) * np.diag(cos2p)

        return H

    ackley = OptFunc(func_expr=ackley_f, vars=list(range(d)),
                     grad_expr=ackley_g, hess_expr=ackley_H)
    funcs["ackley"] = (ackley, np.zeros(d))

    # ---------------------------
    # 3) GRIEWANK (manual)
    # f = sum(xi^2/4000) - prod_i cos(xi/sqrt(i+1)) + 1
    # Sea s_i = sqrt(i+1), θ_i = xi / s_i, P = ∏ cos(θ_i)
    #
    # grad_i = xi/2000 + P * tan(θ_i) / s_i
    #
    # Hessiano:
    #  i == j: H_ii = 1/2000 + P / s_i^2
    #  i != j: H_ij = - P * tan(θ_i) * tan(θ_j) / (s_i * s_j)
    #
    def griewank_f(x):
        x = np.asarray(x, dtype=float)
        idx = np.arange(1, x.size+1, dtype=float)
        s = np.sqrt(idx)
        sum_term = np.sum(x**2) / 4000.0
        prod_term = np.prod(np.cos(x / s))
        return float(sum_term - prod_term + 1.0)

    def griewank_g(x):
        x = np.asarray(x, dtype=float)
        n = x.size
        idx = np.arange(1, n+1, dtype=float)
        s = np.sqrt(idx)
        theta = x / s
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        P = np.prod(cos_theta)

        # grad
        g = x / 2000.0 + P * np.tan(theta) / s
        return g

    def griewank_H(x):
        x = np.asarray(x, dtype=float)
        n = x.size
        idx = np.arange(1, n+1, dtype=float)
        s = np.sqrt(idx)
        theta = x / s
        cos_theta = np.cos(theta)
        P = np.prod(cos_theta)

        H = np.empty((n, n), dtype=float)
        # Diagonal: 1/2000 + P/s_i^2
        diag = 1.0/2000.0 + P / (s**2)
        np.fill_diagonal(H, diag)

        # Fuera de diagonal: - P * tan(theta_i)*tan(theta_j) / (s_i*s_j)
        tan_theta = np.tan(theta)
        H += - P * np.outer(tan_theta / s, tan_theta / s)
        # Sobre-escribimos diagonal (porque la línea anterior también afectó a diagonal)
        np.fill_diagonal(H, diag)
        return H

    griewank = OptFunc(func_expr=griewank_f, vars=list(range(d)),
                       grad_expr=griewank_g, hess_expr=griewank_H)
    funcs["griewank"] = (griewank, np.zeros(d))

    return funcs

funcs = get_test_functions_manual(d=10)

# =====================
# Experimentos
# =====================
desc_methods = ["gradient_descent", "newton", "bfgs"]
step_methods = ["wolfe", "goldstein", "backtracking"]

# Rango para generar puntos iniciales (por función)
ranges = {
    "rosenbrock": (-20, 10),
    "ackley": (-20, 10),
    "griewank": (-600, 300)
}

n_reps = 30  # repeticiones EXACTAS por combinación
max_iter = 1000
tol = 1e-8

# Recoger resultados por función y por combinación
all_summaries = []

for fname, (f, x_true) in funcs.items():
    print(f"\n=== Ejecutando experimentos para: {fname} ===")
    low, high = ranges.get(fname, (-10, 10))

    # Por cada combinación (desc, step) guardamos lista de mse's y fvals
    combo_stats = {}

    # Precrear todas las combinaciones
    for dmethod in desc_methods:
        for smethod in step_methods:
            combo_name = f"{dmethod} + {smethod}"
            combo_stats[combo_name] = {
                "mses": [],
                "fvals": [],
                "successes": 0,
                "errors": []
            }

    # Variables para almacenar la mejor corrida global para esta función
    best_mse = np.inf
    best_info = None  # dict con keys: mse, fval, combo_name, desc, step, x0, x_min, history, trajectory

    # Ejecutar repeticiones
    total_runs = len(desc_methods) * len(step_methods) * n_reps
    pbar = tqdm(total=total_runs, desc=f"Runs {fname}", unit="run")
    for dmethod in desc_methods:
        for smethod in step_methods:
            combo_name = f"{dmethod} + {smethod}"
            for rep in range(n_reps):
                x0 = np.random.uniform(low, high, size=f.n)
                try:
                    ls = LineSearch(function=f,
                                     desc_cond=dmethod,
                                     step_cond=smethod,
                                     x0=x0,
                                     alpha=1.0,
                                     rho=0.5,
                                     c1=1e-4,
                                     c2=0.9,
                                     max_iter=max_iter,
                                     tol=tol)

                    # Pedimos history y trajectory para poder graficar si resulta la mejor corrida
                    res = ls.optimize(return_history=True, return_trajectory=True)

                    # res será (x_min, history, trajectory)
                    if isinstance(res, (tuple, list)) and len(res) >= 3:
                        x_min, history, trajectory = res[0], res[1], res[2]
                    elif isinstance(res, (tuple, list)) and len(res) == 2:
                        # fallback si tu implementación devolviera (x_min, history) o (x_min, trajectory)
                        x_min = res[0]
                        history = res[1]
                        trajectory = np.array([h['x'] for h in history]) if isinstance(history, list) else np.asarray(history)
                    else:
                        x_min = res
                        # si no hay history/trajectory, no podemos plotear esa corrida
                        history = None
                        trajectory = None

                    f_val = f.eval(x_min)
                    mse = float(np.mean((np.asarray(x_min).ravel() - x_true)**2))

                    combo_stats[combo_name]["mses"].append(mse)
                    combo_stats[combo_name]["fvals"].append(f_val)
                    combo_stats[combo_name]["successes"] += 1

                    # Si es la mejor hasta ahora, guardamos datos (incluyendo trajectory)
                    if np.isfinite(mse) and mse < best_mse:
                        best_mse = mse
                        best_info = {
                            "mse": mse,
                            "fval": f_val,
                            "combo_name": combo_name,
                            "desc": dmethod,
                            "step": smethod,
                            "x0": x0.copy(),
                            "x_min": np.asarray(x_min).copy(),
                            "history": history,
                            "trajectory": np.asarray(trajectory) if trajectory is not None else None
                        }

                except Exception as e:
                    combo_stats[combo_name]["mses"].append(np.nan)
                    combo_stats[combo_name]["fvals"].append(np.nan)
                    combo_stats[combo_name]["errors"].append(str(e))

                pbar.update(1)
    pbar.close()

    # Construir resumen por combinación para esta función
    rows = []
    for combo_name, stats in combo_stats.items():
        mses = np.asarray(stats["mses"], dtype=float)
        # ignorar NaNs al calcular media/desviación (si todas son NaN, resultará NaN)
        mean_mse = np.nanmean(mses)
        std_mse = np.nanstd(mses)
        success_rate = stats["successes"] / n_reps
        mean_fval = np.nanmean(np.asarray(stats["fvals"], dtype=float))

        rows.append({
            "combo": combo_name,
            "func": fname,
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "mean_fval": mean_fval,
            "success_rate": success_rate,
            "n_reps": n_reps
        })

    # Ordenar por mean_mse asc (el mejor queda arriba) — si NaN, quedan al final
    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values(by=["mean_mse", "std_mse"], ascending=[True, True], na_position="last")
    all_summaries.append(df_summary)

    # --- Graficar la mejor solución encontrada (si existe)
    if best_info is not None and best_info.get("trajectory") is not None:
        try:
            # Para plotear usamos un LineSearch construido con la misma función;
            # los parámetros desc_cond/step_cond solo sirven para el título y no afectan al plot directamente.
            ls_plot = LineSearch(function=f,
                                 desc_cond=best_info["desc"],
                                 step_cond=best_info["step"],
                                 x0=best_info["x0"],
                                 alpha=1.0,
                                 rho=0.5,
                                 c1=1e-4,
                                 c2=0.9,
                                 max_iter=max_iter,
                                 tol=tol)

            fig, ax, proj = ls_plot.plot_trajectory(history=best_info["history"],
                                                   trajectory=best_info["trajectory"],
                                                   proj_method='pca',    # 'pca' o 'coords'
                                                   components=(0,1),
                                                   grid_points=60,
                                                   padding=0.2,
                                                   show_contours=True,
                                                   levels=40,
                                                   figsize=(8,6))
            fname_fig = f"{fname}_best_trajectory.png"
            fig.savefig(fname_fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Guardada figura de la mejor trayectoria para '{fname}' en '{fname_fig}' (mse={best_info['mse']:.3e}).")
        except Exception as e:
            print(f"No se pudo graficar la trayectoria para '{fname}': {e}")
    else:
        print(f"No se encontró una trayectoria válida para graficar en '{fname}'.")

# Concatenar todos los resúmenes (por función ya ordenados internamente)
final_df = pd.concat(all_summaries, ignore_index=True)

# Reordenar columnas para que la primera sea la 'combo'
final_df = final_df[["combo", "func", "mean_mse", "std_mse", "mean_fval", "success_rate", "n_reps"]]

final_df.to_csv("results_summary.csv", index=False)
print("\nResumen guardado en 'results_summary.csv'.")
print(final_df.head(30))