from LinSearch import LineSearch
from OptFunc import OptFunc
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4")
f_expr = (x1**2 + x2**2 + x3**2 + x4**2)
f = OptFunc(f_expr, [x1, x2, x3, x4])
x0 = np.random.uniform(-1, 1, size=4)

# Crear optimizador
ls = LineSearch(function=f,
                desc_cond='newton',
                step_cond='wolfe',
                x0=x0,
                alpha=1.0,
                rho=0.5,
                c1=1e-4,
                c2=0.9,
                max_iter=200,
                tol=1e-8)

# ejecutar
x_min, history, traj = ls.optimize(return_history=True, return_trajectory=True)
print("x_min =", x_min)
print("f(x_min) =", f.eval(x_min))
print("||grad|| =", np.linalg.norm(f.gradient(x_min)))

# Graficar
ls.plot_trajectory(history=history,
                   trajectory=traj,
                   proj_method='coords',
                   components=(0, 1),
                   grid_points=50,     
                   padding=0.25,
                   show_contours=True,
                   levels=10,
                   figsize=(9,7))

plt.show()