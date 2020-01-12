import matplotlib.pyplot as plt
import math
import numpy as np


def solve_euler_explicit(f, x0, dt, tf, t0=0):
    '''Résout l'équation différentielle dx/dt(t) = f(t, x) à l'aide de la
    méthode d'Euler explicite. Renvoie les listes temps et solution.'''
    T = tf - t0
    nbiter = math.floor(T/dt)
    temps = [i * dt for i in range(nbiter)]
    solution = [x0]
    for i in range(1, nbiter):
        solution.append(solution[i-1] + dt * f(i * dt, solution[i-1]))
    return temps, solution

def convergence_euler(f, x0, tf, t0=0, nbiter=4):
    ''' Pour l'équation différentielle canonique dx/dt = x, retourne le log10
    du maximum des écarts à la solution réelle en fonction du log10 de dt.
    Permet une lecture graphique de l'ordre de convergence du schéma d'Euler
    explicite. Exécute le schéma d'Euler pour dt allant de 0.1 à
    0.1/10**nbiter.'''
    dt0 = 0.1
    list_dt = [dt0/(10**i) for i in range(nbiter)]
    max_difference = []
    for dt in list_dt:
        temps, num_solution = solve_euler_explicit(f, x0, dt, tf, t0)
        num_solution = np.array(num_solution)
        real_solution = np.array([np.exp(t) for t in temps])
        difference = abs(num_solution - real_solution)
        max_error = max(difference)
        max_difference.append(max_error)
    log_list_dt = np.log10(list_dt)
    log_max_difference = np.log10(max_difference)
    return log_list_dt, log_max_difference

# Test performances du solver et illustration de la convergence à l'ordre 1
# du schéma d'Euler explicite.
f = lambda t, x : x
time, num_solution_euler = solve_euler_explicit(f, 1, 0.001, 10)
Y = np.exp(time)
Z = [y - x for y, x in zip(Y, num_solution_euler)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 6))
log_list_dt, log_max_difference_euler = convergence_euler(f, 1, 10)
ax1.plot(time, num_solution_euler, label = 'solution numérique')
ax1.plot(time, Y, label = 'fonction exponentielle')
ax1.plot(time, Z, label = 'difference')
ax1.set_title("Illustration des performances du solver Euler")
ax1.legend()
ax2.plot(log_list_dt, log_max_difference_euler)
ax2.set_title("Illustration de la convergence à l'ordre 1 du schéma d'Euler")
ax2.set(xlabel = 'log10(dt), avec dt le pas de résolution', ylabel = \
    'log10(max écarts à la solution réelle)')
plt.grid(True)
plt.show()

def solve_runge_kutta_ordre_2(f, x0, dt, tf, t0=0):
    ''''Résout l'équation différentielle dx/dt(t) = f(t, x) à l'aide de la
    méthode de Rune_Kutta à l'ordre 2. Renvoie les listes temps et solution.'''
    T = tf - t0
    nbiter = math.floor(T/dt)
    temps = [i * dt for i in range(nbiter)]
    solution = [x0]
    for i in range(1, nbiter):
        F1 = f(i * dt, solution[i-1])
        F2 = f((i+1) * dt, solution[i-1] + dt * F1)
        solution.append(solution[i-1] + dt/2 * (F1 + F2))
    return temps, solution

def convergence_runge_kutta(f, x0, tf, t0=0, nbiter=4):
    ''' Pour l'équation différentielle canonique dx/dt = x, retourne le log10
    du maximum des écarts à la solution réelle en fonction du log10 de dt.
    Permet une lecture graphique de l'ordre de convergence du schéma de
    Runge-Kutta. Exécute le schéma de Runge-Kutta pour dt allant de 0.1 à
    0.1/10**nbiter.'''
    dt0 = 0.1
    list_dt = [dt0/10**i for i in range(nbiter)]
    max_difference = []
    for dt in list_dt:
        temps, num_solution = solve_runge_kutta_ordre_2(f, x0, dt, tf, t0)
        num_solution = np.array(num_solution)
        real_solution = np.array([np.exp(t) for t in temps])
        difference = abs(num_solution - real_solution)
        max_error = max(difference)
        max_difference.append(max_error)
    log_list_dt = np.log10(list_dt)
    log_max_difference = np.log10(max_difference)
    return log_list_dt, log_max_difference

# Test performances du solver Runge-Kutta et comparaison de l'ordre de
# convergence avec le schéma d'Euler explicite.
time, num_solution_runge = solve_runge_kutta_ordre_2(f, 1, 0.001, 10)
Z = [y - x for y, x in zip(Y, num_solution_runge)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 6))
log_list_dt, log_max_difference_runge = convergence_runge_kutta(f, 1, 10)
ax1.plot(time, num_solution_runge, label = 'solution numérique')
ax1.plot(time, Y, label = 'fonction exponentielle')
ax1.plot(time, Z, label = 'difference')
ax1.set_title("Illustration des performances du solver Runge-Kutta (ordre 2)")
ax1.legend()
ax2.plot(log_list_dt, log_max_difference_euler, label = "schéma d'Euler \
    (ordre 1)")
ax2.plot(log_list_dt, log_max_difference_runge, label = 'schéma de Runge-Kutta \
    (ordre 2)')
ax2.set_title("Comparaison des ordres de convergence de deux schémas")
ax2.set(xlabel = 'log10(dt), avec dt le pas de résolution', ylabel = \
    'log10(max écarts à la solution réelle)')
ax2.legend()
plt.grid(True)
plt.show()

def solve_ivp_euler_explicit_variable_step(f, t0, x0, t_f, dtmin=1e-16, dtmax=0.01, atol=1e-6):
    dt = dtmax/10 # initial integration step
    list_dt, list_errors = [], []
    ts, xs = [t0], [x0]  # storage variables
    t = t0
    ti = 0  # internal time keeping track of time since latest storage point : must remain below dtmax
    x = x0
    while ts[-1] < t_f:
        while ti < dtmax:
            t_next, ti_next, x_next = t + dt, ti + dt, x + dt * f(x)
            x_back = x_next - dt * f(x_next)
            ratio_abs_error = atol / (np.linalg.norm(x_back-x)/2)
            dt = 0.9 * dt * math.sqrt(ratio_abs_error)
            if dt < dtmin:
                raise ValueError("Time step below minimum")
            elif dt > dtmax/2:
                dt = dtmax/2
            list_errors.append(atol/ratio_abs_error)
            list_dt.append(dt)
            t, ti, x = t_next, ti_next, x_next
        dt2DT = dtmax - ti # time left to dtmax
        t_next, ti_next, x_next = t + dt2DT, 0, x + dt2DT * f(x)
        ts = np.vstack([ts, t_next])
        xs = np.vstack([xs, x_next])
        t, ti, x = t_next, ti_next, x_next
    return ts, xs, list_dt, list_errors




g = lambda x : x # f ne dépendant pas de t
fig, ax1 = plt.subplots()
t, x, list_dt, list_errors = solve_ivp_euler_explicit_variable_step(g, 0, 1, 10)
color = 'tab:blue'
ax1.set_xlabel('itérations')
ax1.set_ylabel('erreurs', color=color)
ax1.plot(list_errors[1:], color=color)
color = 'tab:red'
ax2 = ax1.twinx()
ax2.plot(list_dt, color=color, linewidth=2.0)
ax2.set_ylabel('dt', color=color)
fig.tight_layout()
plt.show()