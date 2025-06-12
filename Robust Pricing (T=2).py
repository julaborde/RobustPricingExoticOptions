
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import newton
import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import exp1
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


def solve_optimization(G, x_vals, mu_1_n, y_vals, mu_2_n):
    """
    Solves the mixed-integer linear and quadratic optimization problem using Gurobi.
    Returns the optimal value, the matrix of decision variables, and the convergence parameter.
    """
    n = len(x_vals) - 1
    model = gp.Model("Measure Approximation")
    
    # Decision variables p_{i,j}, ensuring non-negativity
    p = model.addVars(n + 1, n + 1, vtype=GRB.CONTINUOUS, lb=0, name="p")
    
    # Objective function: Maximize sum of p_{i,j} * G(x_i, y_j)
    model.setObjective(gp.quicksum(p[i, j] * G(x_vals[i], y_vals[j]) for i in range(n + 1) for j in range(n + 1)), GRB.MAXIMIZE)
    
    # Constraints
    for i in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j] for j in range(n + 1)) <= mu_1_n[i] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j] for j in range(n + 1)) >= mu_1_n[i] - 1e-3)
    for j in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j] for i in range(n + 1)) <= mu_2_n[j] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j] for i in range(n + 1)) >= mu_2_n[j] - 1e-3)
    for i in range(n + 1):
        model.addConstr(gp.quicksum(p[i, j] * y_vals[j] for j in range(n + 1)) <= x_vals[i]*mu_1_n[i] + 1e-3)
        model.addConstr(gp.quicksum(p[i, j] * y_vals[j] for j in range(n + 1)) >= x_vals[i]*mu_1_n[i] - 1e-3)
    
    # Solve the model
    model.optimize()
    
    # Compute the convergence parameter (sum of constraint violations)
    convergence_param = sum(abs(model.getConstrs()[i].slack) for i in range(len(model.getConstrs())))

    if model.status == GRB.OPTIMAL:
        p_matrix = np.array([[p[i, j].x for j in range(n + 1)] for i in range(n + 1)])
        return model.objVal, p_matrix, convergence_param
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

def entropic_mot(x_vals, mu_1_n, y_vals, mu_2_n, cost_function, epsilon=0.1, max_iter=1000, tol=1e-6):
    n = len(x_vals) - 1
    c = np.array([[cost_function(x_vals[i], y_vals[j]) for j in range(n + 1)] for i in range(n + 1)])
    q = np.exp(np.clip(c / epsilon, -100, 100)) # Avoid having arbitrarly large exp()
    p = q / np.sum(q)
    
    convergence = []
    for k in range(1, max_iter + 1):
        p_old = p.copy()
        
        if k % 3 == 1:
            lambda_i = np.log(mu_1_n / np.sum(p, axis=1)) 
            p = p * np.exp(lambda_i[:, None])
        elif k % 3 == 2:
            lambda_j = np.log(mu_2_n / np.sum(p, axis=0))
            p = p * np.exp(lambda_j[None, :])
        else:
            lambda_i = np.array([solve_lambda(y_vals, p[i, :], mu_1_n[i], x_vals[i]) for i in range(n + 1)])
            lambda_i = np.nan_to_num(lambda_i)  # Replace NaNs with 0
            p = p * np.exp(lambda_i[:, None] * y_vals[None, :])
        
        
        p = np.maximum(p, 1e-12)  # Ensure probabilities are non-negative
        p /= np.sum(p)
        convergence.append(np.linalg.norm(p - p_old))
        if convergence[-1] < tol:
            break
    
    optimal_value = np.sum(p * c)
    return optimal_value, p, convergence


def solve_mot(method, x_vals, mu_1_n, y_vals, mu_2_n, cost_function):
    """
    Wrapper function to solve MOT using either Gurobi or entropic regularization.
    """
    mu_1_n = check_and_fix_distribution(mu_1_n)
    mu_2_n = check_and_fix_distribution(mu_2_n)
    
    if method == "gurobi":
        return solve_optimization(cost_function, x_vals, mu_1_n, y_vals, mu_2_n)
    elif method == "entropic":
        return entropic_mot(x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    else:
        raise ValueError("Invalid method. Choose 'gurobi' or 'entropic'.")


def solve_mot_bounds(method, x_vals, mu_1_n, y_vals, mu_2_n, cost_function):
    mu_1_n = check_and_fix_distribution(mu_1_n)
    mu_2_n = check_and_fix_distribution(mu_2_n)
    
    max_value, p_max, conv_max = solve_mot(method, x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    min_value, p_min, conv_min = solve_mot(method, x_vals, mu_1_n, y_vals, mu_2_n, lambda x, y: -cost_function(x, y))
    
    return max_value, -min_value, abs(max_value - min_value), p_max, p_min, conv_max, conv_min

def liquidity_sufficient(upper, lower, threshold=0.05):
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



    
    
def example(method):
    a, b = 1, 3
    a_prime, b_prime = 0, 4
    n = 10
    
    mu = lambda x: 1 / (b - a) if a <= x <= b else 0
    mu_prime = lambda x: 1 / (b_prime - a_prime) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: abs(x - y)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")


def example_exponential(method):
    a, b = 0, 1
    a_prime, b_prime = 1e-3, 3
    n = 100
    
    # X ~ Uniforme[0,1]
    mu = lambda x: 1 if a <= x <= b else 0
    
    integral_val = safe_integrate(exp1, a_prime, b_prime)
    mu_prime = lambda x: exp1(x)/integral_val if a_prime <= x <= b_prime else 0
    
    
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: abs(x - y)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")






def call(method,K):
    a, b = 1, 3
    a_prime, b_prime = 0, 4
    n = 100
    
    mu = lambda x: 1 / (b - a) if a <= x <= b else 0
    mu_prime = lambda x: 1 / (b_prime - a_prime) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: max(0,y-K)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")
    
def put(method,K):
    a, b = 1, 3
    a_prime, b_prime = 0, 4
    n = 100
    
    mu = lambda x: 1 / (b - a) if a <= x <= b else 0
    mu_prime = lambda x: 1 / (b_prime - a_prime) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: max(0,K-y)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")


def asian(method):
    a, b = 1, 3
    a_prime, b_prime = 0, 4
    n = 100
    
    mu = lambda x: 1 / (b - a) if a <= x <= b else 0
    mu_prime = lambda x: 1 / (b_prime - a_prime) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: (x+y)/2
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")


def barrier(method,option,strike_price,param_1,param_2,B):
    a, b = 1, 3
    a_prime, b_prime = 0, 4
    n = 100
    
    mu = lambda x: 1 / (b - a) if a <= x <= b else 0
    mu_prime = lambda x: 1 / (b_prime - a_prime) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    
    k=strike_price
    if option=="call":
        if param_1=="up":
            if param_2=="in":
                cost_function = lambda x, y: max(0,y-k) if max(x,y)>=B else 0
            elif param_2=="out":
                cost_function = lambda x, y: max(0,y-k) if max(x,y)<B else 0
        elif param_1=="down":
            if param_2=="in":
                cost_function = lambda x, y: max(0,y-k) if max(x,y)<=B else 0
            elif param_2=="out":
                cost_function = lambda x, y: max(0,y-k) if max(x,y)>B else 0
    
    elif option=="put":
        if param_1=="up":
            if param_2=="in":
                cost_function = lambda x, y: max(0,k-y) if max(x,y)>=B else 0
            elif param_2=="out":
                cost_function = lambda x, y: max(0,k-y) if max(x,y)<B else 0
        elif param_1=="down":
            if param_2=="in":
                cost_function = lambda x, y: max(0,k-y) if max(x,y)<=B else 0
            elif param_2=="out":
                cost_function = lambda x, y: max(0,k-y) if max(x,y)>B else 0
          
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")

def lookback(method):
    a, b = 1, 3
    a_prime, b_prime = 0, 4
    n = 100
    
    mu = lambda x: 1 / (b - a) if a <= x <= b else 0
    mu_prime = lambda x: 1 / (b_prime - a_prime) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: max(x,y)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")





def asian_gaussian_distribution(method):
    a, b = -5, 5
    a_prime, b_prime = -5, 5
    n = 100
    
    mu = lambda x: norm.pdf(x,1,1) if a <= x <= b else 0
    mu_prime = lambda x: norm.pdf(x,1,1) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: (x+y)/2
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")
    
    
    
def asian_log_normal_distribution(method):
    a, b = np.exp(-5),np.exp(5)
    a_prime, b_prime = np.exp(-5),np.exp(5)
    n = 100
    μ = 1
    σ = 1
    
    mu = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a <= x <= b else 0
    mu_prime = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: (x+y)/2
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")
    
    
    

def lookback_log_normal_distribution(method):
    a, b = np.exp(-5),np.exp(5)
    a_prime, b_prime = np.exp(-5),np.exp(5)
    n = 100
    μ = 1
    σ = 1
    
    mu = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a <= x <= b else 0
    mu_prime = lambda x: lognorm.pdf(x,σ,np.exp(μ)) if a_prime <= x <= b_prime else 0
    
    x_vals, mu_1_n = discretize_measure(mu, a, b, n)
    y_vals, mu_2_n = discretize_measure(mu_prime, a_prime, b_prime, n)
    cost_function = lambda x, y: max(x,y)
    
    if method == "gurobi":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("gurobi", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    if method == "entropic":
        upper_bound, lower_bound, gap, p_max, p_min, conv_max, conv_min = solve_mot_bounds("entropic", x_vals, mu_1_n, y_vals, mu_2_n, cost_function)
    
    print(method)
    print(f"Upper Bound: {upper_bound}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Liquidity Sufficient: {liquidity_sufficient(upper_bound, lower_bound)}")
    
    plot_convergence(conv_max, "Convergence of Entropic Method (Max)")
    plot_convergence(conv_min, "Convergence of Entropic Method (Min)")
    plot_transport_plan(p_max, x_vals, y_vals, "Optimal Transport Plan (Max)")
    plot_transport_plan(p_min, x_vals, y_vals, "Optimal Transport Plan (Min)")

