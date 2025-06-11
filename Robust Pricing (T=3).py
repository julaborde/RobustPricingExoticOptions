#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:22:30 2025

@author: gauthierberanger
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:33:17 2025

@author: gauthierberanger
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import newton
import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm


#Safeguard for distribution
def check_and_fix_distribution(mu):
    if np.any(np.isnan(mu)) or np.sum(mu) == 0:
        print("Warning: Invalid probability distribution detected. Normalizing...")
        mu = np.nan_to_num(mu)  # Replace NaNs with zeros
    mu /= np.sum(mu)  # Renormalize
    return mu


#Safeguard for integration
def safe_integrate(func, a, b):
    """ Safe numerical integration with error handling. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        try:
            result, error = quad(func, a, b)
            return result
        except:
            print(f"Warning: Integration failed for range [{a}, {b}]. Using fallback value.")
            return 0  # Fallback to zero


def discretize_measure(mu, a, b, n):
    """
    Approximate a measure mu with bounded support [a,b] using a discrete measure mu_n.
    """
    x_vals = np.linspace(a, b, n + 1)
    dx = (b - a) / n
    mu_n = np.zeros(n + 1)

    for i in range(n + 1):
        xi = x_vals[i]

        def f1(x): return n * (x - (xi - dx)) * mu(x)
        def f2(x): return n * ((xi + dx) - x) * mu(x)

        integral_1 = safe_integrate(f1, max(a, xi - dx), xi)
        integral_2 = safe_integrate(f2, xi, min(b, xi + dx))
        
        mu_n[i] = integral_1 + integral_2
    
    # Normalize to ensure sum(mu_n) is approximately 1
    mu_n = check_and_fix_distribution(mu_n)

    return x_vals, mu_n


def solve_optimization(G, x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n):
    """
    Solves the mixed-integer linear and quadratic optimization problem using Gurobi.
    Returns the optimal value, the matrix of decision variables, and the convergence parameter.
    """
    n = len(x_vals) - 1
    model = gp.Model("Measure Approximation")
    
    # Decision variables p_{i,j}, ensuring non-negativity
    p = model.addVars(n + 1, n + 1, n+1, vtype=GRB.CONTINUOUS, lb=0, name="p")
    
    # Objective function: Maximize sum of p_{i,j} * G(x_i, y_j)
    model.setObjective(gp.quicksum(p[i, j, k] * G(x_vals[i], y_vals[j], z_vals[k]) for i in range(n + 1) for j in range(n + 1) for k in range(n+1)), GRB.MAXIMIZE)
    
    # Constraints
    for i in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j, k] for j in range(n + 1) for k in range(n+1)) <= mu_1_n[i] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j, k] for j in range(n + 1) for k in range(n+1)) >= mu_1_n[i] - 1e-3)
    for j in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j, k] for i in range(n + 1) for k in range(n+1)) <= mu_2_n[j] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j, k] for i in range(n + 1) for k in range(n+1)) >= mu_2_n[j] - 1e-3)
    for k in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j, k] for i in range(n + 1) for j in range(n+1)) <= mu_3_n[k] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j, k] for i in range(n + 1) for j in range(n+1)) >= mu_3_n[k] - 1e-3)
    for i in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j, k] * y_vals[j] for j in range(n + 1) for k in range(n+1)) <= x_vals[i]*mu_1_n[i] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j, k] * y_vals[j] for j in range(n + 1) for k in range(n+1)) >= x_vals[i]*mu_1_n[i] - 1e-3)
    for j in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j, k] * z_vals[k] for k in range(n + 1) for i in range(n+1)) <= y_vals[j]*mu_2_n[j] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j, k] * z_vals[k] for k in range(n + 1) for i in range(n+1)) >= y_vals[j]*mu_2_n[j] - 1e-3)
    
    # Solve the model
    model.optimize()
    
    # Compute the convergence parameter (sum of constraint violations)
    convergence_param = sum(abs(model.getConstrs()[i].slack) for i in range(len(model.getConstrs())))
    
    
    if model.status == GRB.OPTIMAL:
        p_matrix1 = np.array([[np.sum(p[i, j, k].x for k in range(n+1))for j in range(n + 1)] for i in range(n + 1)])
        p_matrix2 = np.array([[np.sum(p[i, j, k].x for i in range(n+1))for k in range(n + 1)] for j in range(n + 1)])
        return model.objVal, p_matrix1, p_matrix2, convergence_param
    else:
        return None, None, None





def solve_lambda(y_vals, p_row, mu_x_i, x_i):
    def equation(z):
        try:
            return np.sum(y_vals * p_row * np.exp(np.clip(z * y_vals, -50, 50))) - mu_x_i * x_i
        except FloatingPointError:
            return np.inf
    
    try:
        return newton(equation, 0.0, maxiter=100, tol=1e-6)
    except RuntimeError:
        #print(f"Warning: Newton's method did not converge for x_i = {x_i}. Using fallback value.")
        return 0

def entropic_mot(x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function, epsilon=0.1, max_iter=1000, tol=1e-6):
    n = len(x_vals) - 1
    c = np.array([[[cost_function(x_vals[i], y_vals[j], z_vals[k]) for k in range(n+1)] for j in range(n + 1)] for i in range(n + 1)])
    q = np.exp(np.clip(c / epsilon, -100, 100)) # Avoid having arbitrarly large exp()
    p = q / np.sum(q)
    
    convergence = []
    for l in range(1, max_iter + 1):
        p_old = p.copy()
        
        if l % 5 == 1:
            lambda_i = np.log(mu_1_n / np.sum(p, axis=(1,2))) 
            p = p * np.exp(lambda_i[:,None, None])
        elif l % 5 == 2:
            lambda_j = np.log(mu_2_n / np.sum(p, axis=(0,2)))
            p = p * np.exp(lambda_j[None, :, None])
        elif l % 5 == 3:
            lambda_k = np.log(mu_3_n / np.sum(p, axis=(0,1)))
            p = p * np.exp(lambda_k[None, None, :])
        elif l % 5 ==4:
            lambda_i = np.array([solve_lambda(y_vals, p[i, :, :], mu_1_n[i], x_vals[i]) for i in range(n + 1)])
            lambda_i = np.nan_to_num(lambda_i)  # Replace NaNs with 0
            p = p * np.exp(lambda_i[:, None, None] * y_vals[None, :, None])
        else:
            lambda_j = np.array([solve_lambda(z_vals, p[:, j, :], mu_2_n[j], y_vals[j]) for j in range(n + 1)])
            lambda_j = np.nan_to_num(lambda_i)  # Replace NaNs with 0
            p = p * np.exp(lambda_j[None, :, None] * z_vals[None, None, :])
            
        
        
        p = np.maximum(p, 1e-12)  # Ensure probabilities are non-negative
        p /= np.sum(p)
        convergence.append(np.linalg.norm(p - p_old))
        if convergence[-1] < tol:
            break
    
    optimal_value = np.sum(p * c)
    p_matrix1 = np.array([[np.sum(p[i, j, k] for k in range(n+1))for j in range(n + 1)] for i in range(n + 1)])
    p_matrix2 = np.array([[np.sum(p[i, j, k] for i in range(n+1))for k in range(n + 1)] for j in range(n + 1)])
    return optimal_value, p_matrix1, p_matrix2, convergence


def solve_mot(method, x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function):
    """
    Wrapper function to solve MOT using either Gurobi or entropic regularization.
    """
    mu_1_n = check_and_fix_distribution(mu_1_n)
    mu_2_n = check_and_fix_distribution(mu_2_n)
    mu_3_n = check_and_fix_distribution(mu_3_n)
    
    if method == "gurobi":
        return solve_optimization(cost_function, x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n)
    elif method == "entropic":
        return entropic_mot(x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    else:
        raise ValueError("Invalid method. Choose 'gurobi' or 'entropic'.")


def solve_mot_bounds(method, x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function):
    mu_1_n = check_and_fix_distribution(mu_1_n)
    mu_2_n = check_and_fix_distribution(mu_2_n)
    mu_3_n = check_and_fix_distribution(mu_3_n)
    
    max_value, p_max1, p_max2, conv_max = solve_mot(method, x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    min_value, p_min1, p_min2, conv_min = solve_mot(method, x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, lambda x, y, z: -cost_function(x, y, z))
    
    return max_value, -min_value, abs(max_value - min_value), p_max1, p_max2, p_min1, p_min2, conv_max, conv_min

def liquidity_sufficient(upper, lower, threshold=0.1):
    return 2*abs(upper - lower)/(upper + lower) < threshold

def plot_convergence(convergence, title):
    plt.figure()
    plt.plot(convergence, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Difference')
    plt.title(title)
    plt.grid()
    plt.show()

def plot_transport_plan(p, x_vals, y_vals, title):
    plt.figure()
    plt.imshow(p, cmap='viridis', aspect='auto', origin='lower', 
               extent=[y_vals[0], y_vals[-1], x_vals[0], x_vals[-1]])
    plt.colorbar(label='Probability Mass')
    plt.xlabel('y values')
    plt.ylabel('x values')
    plt.title(title)
    plt.show()



    
    
def asian(method):
    a1, b1 = 2, 4
    a2, b2 = 1, 5
    a3, b3 = 0, 6
    n = 100
    
    mu_1 = lambda x: 1 / (b1 - a1) if a1 <= x <= b1 else 0
    mu_2 = lambda x: 1 / (b2 - a2) if a2 <= x <= b2 else 0
    mu_3 = lambda x: 1 / (b3 - a3) if a3 <= x <= b3 else 0
    
    x_vals, mu_1_n = discretize_measure(mu_1, a1, b1, n)
    y_vals, mu_2_n = discretize_measure(mu_2, a2, b2, n)
    z_vals, mu_3_n = discretize_measure(mu_3, a3, b3, n)
    cost_function = lambda x, y, z: (x+y+z)/3
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Max)")
    plot_transport_plan(p_max2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Max)")
    plot_transport_plan(p_min1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Min)")
    plot_transport_plan(p_min2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Min)")
    


def lookback(method):
    a1, b1 = 2, 4
    a2, b2 = 1, 5
    a3, b3 = 0, 6
    n = 100
    
    mu_1 = lambda x: 1 / (b1 - a1) if a1 <= x <= b1 else 0
    mu_2 = lambda x: 1 / (b2 - a2) if a2 <= x <= b2 else 0
    mu_3 = lambda x: 1 / (b3 - a3) if a3 <= x <= b3 else 0
    
    x_vals, mu_1_n = discretize_measure(mu_1, a1, b1, n)
    y_vals, mu_2_n = discretize_measure(mu_2, a2, b2, n)
    z_vals, mu_3_n = discretize_measure(mu_3, a3, b3, n)
    cost_function = lambda x, y, z: max(x,y,z)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Max)")
    plot_transport_plan(p_max2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Max)")
    plot_transport_plan(p_min1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Min)")
    plot_transport_plan(p_min2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Min)")
    
    
def barrier(method,option,strike_price,param_1,param_2,B):
    a1, b1 = 2, 4
    a2, b2 = 1, 5
    a3, b3 = 0, 6
    n = 100
    
    mu_1 = lambda x: 1 / (b1 - a1) if a1 <= x <= b1 else 0
    mu_2 = lambda x: 1 / (b2 - a2) if a2 <= x <= b2 else 0
    mu_3 = lambda x: 1 / (b3 - a3) if a3 <= x <= b3 else 0
    
    x_vals, mu_1_n = discretize_measure(mu_1, a1, b1, n)
    y_vals, mu_2_n = discretize_measure(mu_2, a2, b2, n)
    z_vals, mu_3_n = discretize_measure(mu_3, a3, b3, n)
    
    k=strike_price
    if option=="call":
        if param_1=="up":
            if param_2=="in":
                cost_function = lambda x, y, z: max(0,y-k) if max(x,y,z)>=B else 0
            elif param_2=="out":
                cost_function = lambda x, y, z: max(0,y-k) if max(x,y,z)<B else 0
        elif param_1=="down":
            if param_2=="in":
                cost_function = lambda x, y, z: max(0,y-k) if max(x,y,z)<=B else 0
            elif param_2=="out":
                cost_function = lambda x, y, z: max(0,y-k) if max(x,y,z)>B else 0
    
    elif option=="put":
        if param_1=="up":
            if param_2=="in":
                cost_function = lambda x, y, z: max(0,k-y) if max(x,y,z)>=B else 0
            elif param_2=="out":
                cost_function = lambda x, y, z: max(0,k-y) if max(x,y,z)<B else 0
        elif param_1=="down":
            if param_2=="in":
                cost_function = lambda x, y, z: max(0,k-y) if max(x,y,z)<=B else 0
            elif param_2=="out":
                cost_function = lambda x, y, z: max(0,k-y) if max(x,y,z)>B else 0
          
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Max)")
    plot_transport_plan(p_max2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Max)")
    plot_transport_plan(p_min1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Min)")
    plot_transport_plan(p_min2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Min)")




def asian_gaussian_distribution(method):
    a1, b1 = -5, 5
    a2, b2 = -5, 5
    a3, b3 = -5, 5
    n = 100
    μ = 1
    σ = 1
    
    mu_1 = lambda x: norm.pdf(x,μ,σ) if a1 <= x <= b1 else 0
    mu_2 = lambda x: norm.pdf(x,μ,σ) if a2 <= x <= b2 else 0
    mu_3 = lambda x: norm.pdf(x,μ,σ) if a3 <= x <= b3 else 0
    
    x_vals, mu_1_n = discretize_measure(mu_1, a1, b1, n)
    y_vals, mu_2_n = discretize_measure(mu_2, a2, b2, n)
    z_vals, mu_3_n = discretize_measure(mu_3, a3, b3, n)
    cost_function = lambda x, y, z: (x+y+z)/3
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Max)")
    plot_transport_plan(p_max2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Max)")
    plot_transport_plan(p_min1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Min)")
    plot_transport_plan(p_min2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Min)")


def asian_log_normal_distribution(method):
    a1, b1 = np.exp(-5),np.exp(5)
    a2, b2 = np.exp(-5),np.exp(5)
    a3, b3 = np.exp(-5),np.exp(5)
    n = 100
    μ = 1
    σ = 1
    
    mu_1 = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a1 <= x <= b1 else 0
    mu_2 = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a2 <= x <= b2 else 0
    mu_3 = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a3 <= x <= b3 else 0
    
    x_vals, mu_1_n = discretize_measure(mu_1, a1, b1, n)
    y_vals, mu_2_n = discretize_measure(mu_2, a2, b2, n)
    z_vals, mu_3_n = discretize_measure(mu_3, a3, b3, n)
    cost_function = lambda x, y, z: (x+y+z)/3
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Max)")
    plot_transport_plan(p_max2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Max)")
    plot_transport_plan(p_min1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Min)")
    plot_transport_plan(p_min2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Min)")


def lookback_log_normal_distribution(method):
    a1, b1 = np.exp(-5),np.exp(5)
    a2, b2 = np.exp(-5),np.exp(5)
    a3, b3 = np.exp(-5),np.exp(5)
    n = 100
    μ = 1
    σ = 1
    
    mu_1 = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a1 <= x <= b1 else 0
    mu_2 = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a2 <= x <= b2 else 0
    mu_3 = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a3 <= x <= b3 else 0
    
    x_vals, mu_1_n = discretize_measure(mu_1, a1, b1, n)
    y_vals, mu_2_n = discretize_measure(mu_2, a2, b2, n)
    z_vals, mu_3_n = discretize_measure(mu_3, a3, b3, n)
    cost_function = lambda x, y, z: max(x,y,z)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max1, p_max2, p_min1, p_min2, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, z_vals, mu_3_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Max)")
    plot_transport_plan(p_max2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Max)")
    plot_transport_plan(p_min1, x_vals, y_vals, "Optimal Transport Plan 1->2 (Min)")
    plot_transport_plan(p_min2, y_vals, z_vals, "Optimal Transport Plan 2->3 (Min)")


    
 

