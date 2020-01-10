import matplotlib.pyplot as plt
import math
import numpy as np

def solve_euler_explicit(f, x0, dt, tf, t0=0):
    '''Résout l'équation différentielle dx/dt(t) = f(t, x) à l'aide de la
    méthode d'Euler explicite. Renvoie les listes temps et solution.'''
    T = tf - t0
    iter = math.floor(T/dt)
    temps = [i * dt for i in range(iter)]
    solution = [x0]
    for i in range(1, iter):
        solution.append(solution[i-1] + dt * f(i * dt, solution[i-1]))
    return temps, solution

def convergence_euler(f, x0, tf, t0=0, nbiter=4):
    dt = 1
    list_dt = []
    max_difference = []
    for i in range(nbiter):
        temps, num_solution = solve_euler_explicit(f, x0, dt, tf, t0)
        num_solution = np.array(num_solution)
        real_solution = np.array([np.exp(t) for t in temps])
        difference = abs(num_solution - real_solution)
        max_error = max(difference)
        max_difference.append(max_error)
        list_dt.append(dt)
        dt = dt/10
    log_list_dt = np.log10(list_dt)
    log_max_difference = np.log10(max_difference)
    plt.plot(log_list_dt, log_max_difference)
    plt.show()

def solve_runge_kutta_ordre_2(f, x0, dt, tf, t0=0):
    ''''Résout l'équation différentielle dx/dt(t) = f(t, x) à l'aide de la
    méthode de Rune_Kutta à l'ordre 2. Renvoie les listes temps et solution.'''
    T = tf - t0
    iter = math.floor(T/dt)
    temps = [i * dt for i in range(iter)]
    solution = [x0]
    for i in range(1, iter):
        F1 = f(i * dt, solution[i-1])
        F2 = f((i+1) * dt, solution[i-1] + dt * F1)
        solution.append(solution[i-1] + dt/2 * (F1 + F2))
    return temps, solution

def convergence_runge_kutta(f, x0, tf, t0=0, nbiter=4):
    dt = 1
    list_dt = []
    max_difference = []
    for i in range(nbiter):
        temps, num_solution = solve_runge_kutta_ordre_2(f, x0, dt, tf, t0)
        num_solution = np.array(num_solution)
        real_solution = np.array([np.exp(t) for t in temps])
        difference = abs(num_solution - real_solution)
        max_error = max(difference)
        max_difference.append(max_error)
        list_dt.append(dt)
        dt = dt/10
    log_list_dt = np.log10(list_dt)
    log_max_difference = np.log10(max_difference)
    plt.plot(log_list_dt, log_max_difference)
    plt.show()

## Test solve_euler_explicit avec exp.
dt_exp = 0.001
f = lambda t, x : x
# x peut être un vecteur de dimension n>1 sans problème, seul l'affichage pose
# problème.
time, num_solution = solve_euler_explicit(f, 1, dt_exp, 10)
Y = np.exp(time)
Z = [Y[i] - num_solution[i] for i in range(len(num_solution))]
plt.plot(time, num_solution, label = 'num')
plt.plot(time, Y, label = 'exp')
plt.plot(time, Z, label = 'difference')
plt.legend()
plt.show()


