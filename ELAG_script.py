import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# Symbolic definitions
g, R, l = sp.symbols('g R l', positive=True)
psi_0, psi_dot_0, phi_dot_0, theta_dot_0 = sp.symbols('psi_0 psi_dot_0 phi_dot_0 theta_dot_0', real=True)
psi, psi_dot, phi_dot, theta_dot = sp.symbols('psi psi_dot phi_dot theta_dot', real=True)

# Positions and velocitys
h = -l * sp.sin(psi)
r_s = sp.Matrix([l, 0, 0])
omega = sp.Matrix([theta_dot - sp.sin(psi) * phi_dot, psi_dot, sp.cos(psi) * phi_dot])
Traegheitsmoment_s = 0.25 * R**2 * sp.Matrix([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
v_s = omega.cross(r_s)

# Energies
E_pot = g * h
E_kin = 0.5 * v_s.dot(v_s) + 0.5 * omega.dot(Traegheitsmoment_s * omega)
Lagrangian = E_kin - E_pot

# Partial derivatives
L_psi = sp.diff(Lagrangian, psi_dot)
L_phi = sp.diff(Lagrangian, phi_dot)
L_theta = sp.diff(Lagrangian, theta_dot)

# Substitute initial conditions
L_phi_0 = L_phi.subs({psi: psi_0, phi_dot: phi_dot_0, theta_dot: theta_dot_0})
L_theta_0 = L_theta.subs({psi: psi_0, phi_dot: phi_dot_0, theta_dot: theta_dot_0})

# Solve for phi_dot and theta_dot symbolically
sym_sol = sp.solve([sp.Eq(L_phi, L_phi_0), sp.Eq(L_theta, L_theta_0)], [phi_dot, theta_dot], dict=True)
phi_dot_sym = sym_sol[0][phi_dot]
theta_dot_sym = sym_sol[0][theta_dot]

# Psi equation of motion
DLagrangian_Dpsi = sp.diff(Lagrangian, psi)
DLagrangian_Ddpsi_dt = sp.symbols('DLagrangian_Ddpsi_dt')
Drall_psi = sp.Eq(L_psi, DLagrangian_Ddpsi_dt)
psi_ddot = sp.solve(Drall_psi, psi_dot)[0]
psi_ddot_sym = psi_ddot.subs(DLagrangian_Ddpsi_dt, DLagrangian_Dpsi.subs({phi_dot: phi_dot_sym, theta_dot: theta_dot_sym}))

# -----------------------------
# Parameters
# -----------------------------
g0 = 9.81 # m/s^2
R0 = 0.125 # m
l0 = R0 # m
psi_00 = 0 # rad
psi_dot_00 = 0 # rad/s
phi_dot_00 = 0 # rad/s
theta_dot_00 = 10 * 2 * np.pi # rad/s
T = 1 # total time in seconds

param_subs = {g: g0, R: R0, l: l0,
              psi_0: psi_00, psi_dot_0: psi_dot_00,
              phi_dot_0: phi_dot_00, theta_dot_0: theta_dot_00}

# Subsitute parameters into symbolic expressions for psi_ddot, phi_dot, and theta_dot
psi_ddot_p = psi_ddot_sym.subs(param_subs)
phi_dot_p = phi_dot_sym.subs(param_subs)
theta_dot_p = theta_dot_sym.subs(param_subs)

# Convert symbolic functions to numerical functions
psi_ddot_num = sp.lambdify(psi, psi_ddot_p, 'numpy')
phi_dot_num = sp.lambdify(psi, phi_dot_p, 'numpy')
theta_dot_num = sp.lambdify(psi, theta_dot_p, 'numpy')

# -----------------------------
# ODE system
# -----------------------------
def ode(t, y):
    psi_val = y[0]
    return [y[1],
            psi_ddot_num(psi_val),
            phi_dot_num(psi_val),
            theta_dot_num(psi_val)]

y0 = [psi_00, psi_dot_00, 0, 0]
t_span = (0, T)

ode_sol = solve_ivp(ode, t_span, y0, t_eval=np.linspace(0, T, 500))

t = ode_sol.t
psi_sol = ode_sol.y[0]
psi_dot_sol = ode_sol.y[1]
phi_sol = ode_sol.y[2]
theta_sol = ode_sol.y[3]
phi_dot_sol = phi_dot_num(psi_sol)
theta_dot_sol = theta_dot_num(psi_sol)

# -----------------------------
# Plots
# -----------------------------
const = 0.5 * np.pi * np.ones_like(t)

# Psi plot
plt.figure(figsize=(7,4))
plt.plot(t, psi_sol, linewidth=2)
plt.plot(t, psi_dot_sol, linewidth=2)
plt.plot(t, const, 'r')
plt.xlabel('t [s]')
plt.ylabel(r'$\psi(t)\ [\mathrm{rad}],\  \dot{\psi}(t)\ [\mathrm{rad/s}]$')
plt.title(r'Nutation')
plt.legend([r'$\psi(t)$', r'$\dot{\psi}(t)$', r'$\psi = \frac{\pi}{2}$'], loc='best')
plt.grid(True)

# Phi plot
plt.figure(figsize=(7,4))
plt.plot(t, phi_sol, linewidth=2)
plt.plot(t, phi_dot_sol, linewidth=2)
plt.plot(t, 2*const, 'r')
plt.xlabel('t [s]')
plt.ylabel(r'$\phi(t)\ [\mathrm{rad}],\  \dot{\phi}(t)\ [\mathrm{rad/s}]$')
plt.title(r'Precession')
plt.legend([r'$\phi(t)$', r'$\dot{\phi}(t)$', r'$\phi = \pi$'], loc='best')
plt.grid(True)

# Theta plot
plt.figure(figsize=(7,4))
plt.plot(t, theta_sol, linewidth=2)
plt.plot(t, theta_dot_sol, linewidth=2)
plt.plot(t, 4*const, 'r')
plt.xlabel('t [s]')
plt.ylabel(r'$\theta(t)\ [\mathrm{rad}],\  \dot{\theta}(t)\ [\mathrm{rad/s}]$')
plt.title(r'Spin')
plt.legend([r'$\theta(t)$', r'$\dot{\theta}(t)$', r'$\theta = 2\pi$'], loc='best')
plt.grid(True)

plt.show()