import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import exp1
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import lognorm

def calcul_q(x, y, cost_func, epsilon):
    """Calcule q_ij = exp(G(x_i, y_j) / epsilon)"""
    m, n = len(x), len(y)
    q = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            q[i, j] = np.exp(cost_func(x[i], y[j]) / epsilon) # cost_func(x[i], y[j]) est de l'ordre de l'unité généralement
            
    return q

def calculer_objectif(p, q, epsilon):
    """Calcule la valeur de la fonction objectif L(p) + epsilon * E(p)"""
    # Calculer KL(p|q) pour les valeurs non nulles
    mask = (p > 1e-10) & (q > 1e-10)
    kl_div = np.sum(p[mask] * (np.log(p[mask] / q[mask])))
    
    return epsilon * kl_div

def calculer_prix(p, x, y, cost_func):
    
    G_matrix = np.array([[cost_func(xi, yj) for yj in y] for xi in x])
    expected_price = np.sum(p * G_matrix)
    
    return(np.abs(expected_price))
    
    
    
def update_C1(p, mu):
    m, n = p.shape
    p_next = np.zeros((m, n))
    for i in range(m):
        row_sum = np.sum(p[i, :])
        if row_sum > 1e-10: #éviter la division par 0
            p_next[i, :] = mu[i] * p[i, :] / row_sum
        else:
            p_next[i, :] = 0.0  # La somme est nulle donc tous les p_i,j le sont aussi
    return p_next


def update_C2(p, nu):
    """Mise à jour pour la contrainte C2 (somme sur i de p_ij = beta_j)
    
    Formule : p_2_i,j = beta_j * p_1_i,j / Somme des k allant de 1 à m des p_1_k,j
    """
    m, n = p.shape
    p_next = np.zeros((m, n))
    for j in range(n):
        col_sum = np.sum(p[:, j])
        if col_sum > 1e-10:
            p_next[:, j] = nu[j] * p[:, j] / col_sum
        else:
            p_next[:, j] = 0.0  # La somme est nulle donc tous les p_i,j le sont aussi
            
    return p_next

def solve_lambda_equation_for_martingale(i, p, mu, x, y, tolerance=1e-6, max_iter=100):
    """
    Résout l'équation pour lambda_i dans la contrainte de martingale
    en utilisant la méthode de Newton-Raphson.
    
    D'après l'énoncé, on cherche λ_i tel que :
    alpha_i * x_i = Somme des j allant de 1 à n de (y_j * p_i,j * exp(lambda_i * y_j))
    """
    p_i = p[i, :]
    x_i = x[i]
    alpha_i = mu[i]
    
    # Fonction dont on cherche la racine:
    # f(lambda) = Somme(y_j * p_i,j * exp(lambda * y_j)) - alpha_i * x_i
    def f(lambda_val):
        result = 0
        for j in range(len(y)):
            result += y[j] * p_i[j] * np.exp(np.clip(lambda_val * y[j], -30, 30))
        return result - alpha_i * x_i
    

    # Dérivée de f par rapport à lambda:
    # f'(lambda) = Somme(y_j^2 * p_i,j * exp(lambda * y_j))
    def df(lambda_val):
        result = 0
        for j in range(len(y)):
            result += (y[j]**2) * p_i[j] * np.exp(np.clip(lambda_val * y[j], -30, 30))
        return result
    
    # Point de départ (initial guess)
    lambda_i = 0.0
    
    # Itération de Newton-Raphson
    for iter in range(max_iter):
        f_val = f(lambda_i)
        
        # Vérifier si on a déjà convergé
        if abs(f_val) < tolerance:  # On a trouvé le lambda qu'il faut quand f vaut 0
            return lambda_i
        
        df_val = df(lambda_i)
        
        # Éviter la division par zéro ou valeurs très petites
        if abs(df_val) < 1e-10:
            # Si la dérivée est presque nulle, faire un petit pas
            lambda_i += 0.01 * (1 if f_val > 0 else -1)
        else:
            # Mise à jour de Newton-Raphson
            lambda_i = lambda_i - f_val / df_val
    
    return lambda_i



def update_C3(p, mu, x, y, tolerance = 1e-6):
    """Mise à jour pour la contrainte C3 (martingale: E[Y|X=x_i] = x_i)
    
    D'après l'énoncé:
    - alpha_i * x_i = Somme des j allant de 1 à n de (y_j * p_i,j * exp(lambda_i * y_j))
    - p_3_i,j = p_2_i,j * exp(lambda_i * y_j)
    """
    m, n = p.shape
    p_next = np.zeros((m, n))
    
    for i in range(m):      
        # Trouver lambda_i qui satisfait la contrainte de martingale
        lambda_i = solve_lambda_equation_for_martingale(i, p, mu, x, y, tolerance)
        
        # Mettre à jour p_ij en utilisant lambda_i selon la formule fournie:
        # p_3_i,j = p_2_i,j * exp(lambda_i * y_j)
        for j in range(n):
            p_next[i, j] = p[i, j] * np.exp(np.clip(lambda_i * y[j], -30, 30))
    
    return p_next




def verifier_contraintes(p, mu, nu, x, y):
    """
    Vérifie les trois contraintes et retourne leurs violations
    Retourne un tuple avec (violation maximale, violation mu, violation nu, violation martingale)
    """
    # Contraintes marginales
    mu_constraint = np.abs(p.sum(axis=1) - mu).max()
    nu_constraint = np.abs(p.sum(axis=0) - nu).max()
    
    # Contrainte de martingale
    mart_constraint = 0
    m = len(mu)
    for i in range(m):
        if mu[i] > 1e-10:
            conditional_expectation = np.sum(y * p[i, :]) / mu[i]
            mart_constraint = max(mart_constraint, np.abs(conditional_expectation - x[i]))
    
    # Retourner la violation maximale et les violations individuelles
    max_violation = max(mu_constraint, nu_constraint, mart_constraint)
    
    return max_violation, mu_constraint, nu_constraint, mart_constraint




def resoudre_mot(mu, nu, x, y, cost_func, N_iterations, epsilon=0.1, tol=1e-6, tolerance = 1e-6, verbose=True):
    """
    Résout le problème de transport optimal de martingale.
    
    Paramètres:
    -----------
    mu : array-like - Distribution marginale source (alpha_i)
    nu : array-like - Distribution marginale cible (beta_j)
    x : array-like - Points de support de mu
    y : array-like - Points de support de nu
    cost_func : function - Fonction de coût G(x_i, y_j)
    N_iterations : int - Nombre maximal d'itérations
    epsilon : float - Paramètre de régularisation entropique
    tol : float - Tolérance pour le critère de convergence
    
    Retourne:
    ---------
    p : ndarray - La matrice de transport optimale
    historique : dict - L'historique des violations de contraintes
    """
    # Convertir en numpy arrays
    mu, nu = np.array(mu), np.array(nu)
    x, y = np.array(x), np.array(y)
    
    
    # Initialisation: q_ij = exp(G(x_i, y_j) / epsilon) et p_0 = q
    
    if init_q :
        q = calcul_q(x, y, cost_func, epsilon)
        p = q.copy()
    
    else : 
        p = np.ones((m, n)) / (m * n)
        
    
    # Historique des itérations
    historique = {
        'violations_max': [], 
        'violations_mu': [],
        'violations_nu': [],
        'violations_mart': [],
        'objectifs': []
    }
    
    print(f"Début de la résolution avec {N_iterations} itérations maximales")
    print(f"{'Iter':>5} | {'Contrainte':>10} | {'Violation μ':>12} | {'Violation ν':>12} | {'Violation mart':>12} | {'Prix':>12} | {'Temps (s)':>9}")
    print("-" * 80)
        
    start_time = time.time()
    
    for k in range(1, N_iterations + 1):
        # Déterminer quelle contrainte appliquer dans cette itération
        constraint_idx = (k - 1) % 3 + 1
        constraint_name = {1: "C1 (mu)", 2: "C2 (nu)", 3: "C3 (mart)"}[constraint_idx]
        
        if constraint_idx == 1:
            p = update_C1(p, mu)
        elif constraint_idx == 2:
            p = update_C2(p, nu)
        elif constraint_idx == 3:
            p = update_C3(p, mu, x, y, tolerance)
        
        # Vérifier les contraintes
        max_viol, mu_viol, nu_viol, mart_viol = verifier_contraintes(p, mu, nu, x, y)
        historique['violations_max'].append(max_viol)
        historique['violations_mu'].append(mu_viol)
        historique['violations_nu'].append(nu_viol)
        historique['violations_mart'].append(mart_viol)
        
        # Calculer la valeur de la fonction objectif
        objectif = None
        if k % 10 == 0 or k == 1 or k == N_iterations:
            objectif = calculer_objectif(p, q, epsilon)
            #objectif = calculer_prix(p, x, y, cost_func)
            historique['objectifs'].append((k, objectif))
            
        # Afficher progression
        if verbose and (k % 10 == 0 or k == 1):
            elapsed = time.time() - start_time
            obj_str = f"{objectif:.6f}" if objectif is not None else "---"
            print(f"{k:5d} | {constraint_name:10s} | {mu_viol:.6e} | {nu_viol:.6e} | {mart_viol:.6e} | {obj_str:12s} | {elapsed:.2f}")
        
        # Vérifier la convergence
        if max_viol < tol:
            if verbose:
                elapsed = time.time() - start_time
                print("-" * 80)
                print(f"Convergence atteinte après {k} itérations en {elapsed:.2f} secondes.")
                print(f"Violations finales: μ = {mu_viol:.6e}, ν = {nu_viol:.6e}, mart = {mart_viol:.6e}")
            break
    
    end_time = time.time()
    if verbose and k == N_iterations:
        print("-" * 80)
        print(f"Nombre maximum d'itérations ({N_iterations}) atteint sans convergence.")
        print(f"Temps d'exécution total: {end_time - start_time:.2f} secondes")
    
    return p, historique



def calculer_objectif(p, q, epsilon):
    """Calcule la valeur de la fonction objectif L(p) + epsilon * E(p)"""
    
    # Vérifier si q contient des valeurs infinies
    if np.any(np.isinf(q)):
        indices_inf = np.where(np.isinf(q))
        raise ValueError(f"La matrice q contient des valeurs infinies aux indices {indices_inf}. "
                        f"Valeur maximale de q (hors inf): {np.max(q[~np.isinf(q)])}")
    
    # Calculer KL(p|q) pour les valeurs non nulles
    mask = (p > 1e-10) & (q > 1e-10)  # Permet d'éviter que p ou q soit nulle ce qui pourrait engendrer des erreurs de calcul
    kl_div = np.sum(p[mask] * (1 - np.log(p[mask] / q[mask])))
    
    return epsilon * kl_div # On regarde que kl_div plutot que epsilon * kl_div car pour des valeurs très faibles de epsilon on observe pas l'évolution de la fonction objectif
    
    
    

def afficher_matrice_transport(p, x, y, n_cols_texte=0):
    """
    Affiche une matrice de transport sous forme de heatmap avec contraste optimisé.
    Option d'afficher les fonctions ksi+ et ksi- calculées numériquement.
    
    Paramètres:
    -----------
    p : ndarray - Matrice de transport
    x : array - Points sources
    y : array - Points cibles
    normalisation : str - Type de normalisation : 'globale' ou 'colonne'
    n_cols_texte : int - Affiche les valeurs des maxima toutes les n colonnes (0 pour aucun)
    upper : bool - Indique s'il s'agit de la borne supérieure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    
    if upper : 
        titre= "Matrice de Transport Optimal (Borne sup)"
    else :
        titre= "Matrice de Transport Optimal (Borne inf)"
    
    # Étape 1: Préparation des données
    # Trier les indices par valeurs croissantes de x et y
    x_indices = np.argsort(x)
    
    y_indices = np.argsort(y)  # Déjà correct si y est numérique !
    y_sorted = y[y_indices]

    
    # Réorganiser la matrice selon ces indices triés
    p_sorted = p[np.ix_(x_indices, y_indices)]
    x_sorted = x[x_indices]
    
    
    # Dimensions
    m, n = p_sorted.shape
    
    # Étape 2: Identifier les maxima par colonne
    col_max_indices = np.argmax(p_sorted, axis=0)
    col_max_values = np.array([p_sorted[col_max_indices[j], j] for j in range(n)])
    
    # Valeurs globales
    global_max = np.max(p_sorted)
    mean_max = np.mean(col_max_values[col_max_values > 0]) if np.any(col_max_values > 0) else 0
    min_val = np.min(p_sorted[p_sorted > 0]) if np.any(p_sorted > 0) else 0
    
    # Seuil minimal pour filtrer les valeurs trop faibles (0.1% du max global)
    min_threshold = 0.001 * global_max
    
    # Étape 3: Normalisation de la matrice selon le mode choisi
    norm_matrix = np.zeros_like(p_sorted)
    
    if normalisation == "globale":
        # Pour la normalisation globale, on garde les valeurs d'origine au-dessus du seuil
        # pour pouvoir utiliser une normalisation à deux pentes (TwoSlopeNorm)
        mask = p_sorted > min_threshold
        norm_matrix[mask] = p_sorted[mask]
        
        # Définir les limites pour la normalisation bicolore
        # Centre sur la moyenne des maxima, limites à ±10x cette moyenne
        vmin = max(min_threshold, mean_max / 10)  # au moins le seuil minimal
        vcenter = mean_max
        vmax = min(global_max, mean_max * 10)  # au plus le maximum global
        
        # Normalisation à deux pentes pour la visualisation
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:  # normalisation == "colonne"
        # Normalisation par colonne - proportionnelle au maximum de chaque colonne
        for j in range(n):
            col_max = col_max_values[j]
            if col_max > 0:
                # Filtre et normalisation
                mask = p_sorted[:, j] > min_threshold
                norm_matrix[mask, j] = p_sorted[mask, j] / col_max
        
        # Normalisation standard pour la visualisation en mode colonne
        norm = None
    
    # Étape 4: Préparation de la visualisation
    # Colormap personnalisée selon le mode de normalisation
    if normalisation == "globale":
        # Colormap avec deux gammes de couleurs distinctes: bleu pour <moyenne, rouge pour >moyenne
        colors_below = [
            (0.9, 0.95, 1.0),   # Bleu très clair (valeurs faibles)
            (0.7, 0.85, 1.0),   # Bleu clair
            (0.5, 0.7, 0.95),   # Bleu moyen
            (0.2, 0.4, 0.9),    # Bleu
            (0.0, 0.2, 0.8)     # Bleu foncé (proche de la moyenne)
        ]
        
        colors_above = [
            (0.8, 0.0, 0.0),    # Rouge foncé (proche de la moyenne)
            (0.9, 0.2, 0.0),    # Rouge
            (1.0, 0.4, 0.0),    # Rouge-orange
            (1.0, 0.7, 0.0),    # Orange
            (1.0, 1.0, 0.0)     # Jaune (valeurs élevées)
        ]
        
        # Créer une colormap divergente
        cmap = LinearSegmentedColormap.from_list(
            'diverging_cmap', 
            colors_below + colors_above,
            N=256
        )
    else:
        # Colormap standard pour le mode colonne
        colors = [
            (1.00, 1.00, 1.00, 0.00),  # Blanc transparent (0%)
            (0.90, 0.95, 1.00, 0.50),  # Bleu très clair (15%)
            (0.65, 0.80, 0.95, 0.70),  # Bleu clair (30%)
            (0.40, 0.65, 0.90, 0.80),  # Bleu moyen (50%)
            (0.20, 0.45, 0.85, 0.90),  # Bleu (70%)
            (0.10, 0.20, 0.70, 0.95),  # Bleu foncé (80%)
            (0.30, 0.00, 0.70, 0.95),  # Violet (85%)
            (0.70, 0.00, 0.20, 1.00),  # Rouge (90%)
            (0.90, 0.40, 0.00, 1.00),  # Orange (95%)
            (1.00, 0.80, 0.00, 1.00),  # Jaune-orange (98%)
            (1.00, 1.00, 0.00, 1.00),  # Jaune (100%)
        ]
        
        positions = [0, 0.15, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
        cmap = LinearSegmentedColormap.from_list('transport_cmap', list(zip(positions, colors)), N=256)
    
    # Étape 5: Création de la figure
    aspect_ratio = n / m
    fig_width = min(12, max(8, 8 * aspect_ratio))
    fig_height = min(10, max(6, 8 / aspect_ratio))
    
    plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # Afficher la matrice
    if normalisation == "globale":
        # En mode global, on utilise la matrice originale et une normalisation à deux pentes
        im = plt.imshow(norm_matrix, interpolation='nearest', aspect='auto',
                      cmap=cmap, norm=norm, origin='lower')
    else:
        # En mode colonne, on utilise la matrice normalisée
        im = plt.imshow(norm_matrix, interpolation='nearest', aspect='auto',
                      cmap=cmap, vmin=0, vmax=1.0, origin='lower')
    
    # S'assurer que toute la matrice est visible
    plt.xlim(-0.5, n-0.5)
    plt.ylim(-0.5, m-0.5)
    
    # Étape 6: Personnalisation de l'affichage
    # Ajout des axes et légendes
    if m > 20 or n > 20:
        max_ticks = 15
        x_step = max(1, n // max_ticks)
        y_step = max(1, m // max_ticks)
        
        plt.xticks(range(0, n, x_step), [f"{y_sorted[i]:.2f}" for i in range(0, n, x_step)], rotation=45)
        plt.yticks(range(0, m, y_step), [f"{x_sorted[i]:.2f}" for i in range(0, m, y_step)])
    else:
        plt.xticks(range(n), [f"{y_sorted[i]:.2f}" for i in range(n)], rotation=45)
        plt.yticks(range(m), [f"{x_sorted[i]:.2f}" for i in range(m)])
    
    # Colorbar avec valeurs numériques pour chaque couleur
    cbar = plt.colorbar(im, label='Valeur numérique')
    
    if normalisation == "globale":
        # Modification pour la normalisation globale - afficher les valeurs numériques
        # Créer une échelle logarithmique pour les valeurs entre vmin et vmax
        tick_values = [
            vmin,
            (vmin + vcenter) / 4,
            (vmin + vcenter) / 2,
            vcenter,
            (vcenter + vmax) / 3,
            (vcenter + vmax) / 2,
            vmax
        ]
        
        # Labels indiquant les valeurs numériques exactes
        tick_labels = [
            f"{vmin:.2e}",
            f"{(vmin + vcenter) / 4:.2e}",
            f"{(vmin + vcenter) / 2:.2e}",
            f"{vcenter:.2e}",
            f"{(vcenter + vmax) / 3:.2e}",
            f"{(vcenter + vmax) / 2:.2e}",
            f"{vmax:.2e}"
        ]
        
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels)
    else:
        # Ticks adaptés pour le mode colonne - montrer les valeurs réelles correspondantes
        tick_positions = [0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
        
        # Calculer les valeurs moyennes correspondant à chaque niveau de pourcentage
        col_values = []
        for pos in tick_positions:
            if pos == 0:
                col_values.append(0)
            else:
                # Trouver les valeurs correspondant à chaque niveau de pourcentage pour chaque colonne
                all_vals = []
                for j in range(n):
                    col_max = col_max_values[j]
                    if col_max > 0:
                        val = col_max * pos
                        all_vals.append(val)
                
                # Moyenne des valeurs si disponible
                val = np.mean(all_vals) if all_vals else 0
                col_values.append(val)
        
        # Étiquettes combinant pourcentage et valeur réelle
        tick_labels = [
            f"0% (0)",
            f"15% ({col_values[1]:.2e})",
            f"30% ({col_values[2]:.2e})",
            f"50% ({col_values[3]:.2e})",
            f"70% ({col_values[4]:.2e})",
            f"85% ({col_values[5]:.2e})",
            f"95% ({col_values[6]:.2e})",
            f"100% ({col_values[7]:.2e})"
        ]
        
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
    
    # Étape 7: Marquage des maxima par colonne
    max_points_x = []
    max_points_y = []
    
    for j in range(n):
        i = col_max_indices[j]
        val = p_sorted[i, j]
        if val > min_threshold:
            max_points_y.append(i)
            max_points_x.append(j)
    
    # Afficher les points
    if max_points_x:
        plt.scatter(max_points_x, max_points_y, s=30, c='black', marker='o', 
                   alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # Étape 8: Affichage des valeurs des maxima si demandé
    if n_cols_texte > 0:
        interval = max(1, n // n_cols_texte)
        
        for j in range(n):
            if j % interval == 0:
                i = col_max_indices[j]
                val = p_sorted[i, j]
                if val > min_threshold:
                    plt.text(j, i, f"{val:.2e}", ha='center', va='bottom', 
                            color='black', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', 
                                     ec='none', alpha=0.8))
    
    if show_xi_functions and exemple == "uniforme_exponentielle" and upper :
        """"def E1(y):
            return exp1(y)

        # Calcul de a(x) = ∫_0^x sqrt(1 - 1/E1(t)) dt
        def a(x):
            integral, _ = quad(lambda t: np.sqrt(1 - 1/E1(t)) if E1(t) > 1 else 0, 0, x)
            return integral"""

        # Calcul des fonctions xi_plus et xi_minus
        x_values = np.linspace(0.0001, 1, 1000)  # Éviter x=0 à cause de la singularité

        from a_x_log_normal import a_values
        
        xi_plus = [x_values[i] + a_values[i] for i in range(1000)]
        xi_minus = [x_values[i] - a_values[i] for i in range(1000)]
        
        #xi_plus = [x + a(x) for x in x_values]
        #xi_minus = [x - a(x) for x in x_values]

        # Conversion des valeurs x et y pour correspondre à l'échelle du graphique
        x_scale = n / (max(y_sorted) - min(y_sorted))
        y_scale = m / (max(x_sorted) - min(x_sorted))
        
        xi_plus_scaled = [(xi - min(y_sorted)) * x_scale for xi in xi_plus]
        xi_minus_scaled = [(xi - min(y_sorted)) * x_scale for xi in xi_minus]
        x_values_scaled = [(x - min(x_sorted)) * y_scale for x in x_values]
        
        # Tracer les courbes
        plt.plot(xi_plus_scaled, x_values_scaled, label=r"$\xi_+(x)$", color="blue", linewidth=2)
        plt.plot(xi_minus_scaled, x_values_scaled, label=r"$\xi_-(x)$", color="red", linewidth=2)
        plt.legend(loc='best')
        
    
    if show_xi_functions and exemple == "uniforme_uniforme" and upper :

        # Calcul des fonctions xi_plus et xi_minus
        x_values = np.linspace(1, 3, 1000)  # Éviter x=0 à cause de la singularité
        xi_plus = [x + 1 for x in x_values]
        xi_minus = [x - 1 for x in x_values]

        # Conversion des valeurs x et y pour correspondre à l'échelle du graphique
        x_scale = n / (max(y_sorted) - min(y_sorted))
        y_scale = m / (max(x_sorted) - min(x_sorted))
        
        xi_plus_scaled = [(xi - min(y_sorted)) * x_scale for xi in xi_plus]
        xi_minus_scaled = [(xi - min(y_sorted)) * x_scale for xi in xi_minus]
        x_values_scaled = [(x - min(x_sorted)) * y_scale for x in x_values]
        
        # Tracer les courbes
        plt.plot(xi_plus_scaled, x_values_scaled, label=r"$\xi_+(x)$", color="blue", linewidth=2)
        plt.plot(xi_minus_scaled, x_values_scaled, label=r"$\xi_-(x)$", color="red", linewidth=2)
        plt.legend(loc='best')
        
    if show_xi_functions and exemple == "gaussiennes_decalees" and upper :

        # Calcul des fonctions xi_plus et xi_minus
        x_values = np.linspace(0, 6, 1000)  # Éviter x=0 à cause de la singularité
        xi_plus = [x + np.sqrt(3) for x in x_values]
        xi_minus = [x - np.sqrt(3) for x in x_values]

        # Conversion des valeurs x et y pour correspondre à l'échelle du graphique
        x_scale = n / (max(y_sorted) - min(y_sorted))
        y_scale = m / (max(x_sorted) - min(x_sorted))

        
        xi_plus_scaled = [(xi - min(y_sorted)) * x_scale for xi in xi_plus]
        xi_minus_scaled = [(xi - min(y_sorted)) * x_scale for xi in xi_minus]
        x_values_scaled = [(x - min(x_sorted)) * y_scale for x in x_values]
        
        # Tracer les courbes
        plt.plot(xi_plus_scaled, x_values_scaled, label=r"$\xi_+(x)$", color="blue", linewidth=2)
        plt.plot(xi_minus_scaled, x_values_scaled, label=r"$\xi_-(x)$", color="red", linewidth=2)
        plt.legend(loc='best')
        

        
    # Finalisation
    plt.title(titre, fontsize=14)
    plt.xlabel('Points cibles y_j', fontsize=12)
    plt.ylabel('Points sources x_i', fontsize=12)
    
    # Informations supplémentaires
    norm_info = "globale (par rapport à la moyenne)" if normalisation == "globale" else "par colonne"
    plt.figtext(0.01, 0.01, f"Max: {global_max:.2e} | Moy. max: {mean_max:.2e}", 
               fontsize=8)
    plt.figtext(0.99, 0.01, f"Normalisation: {norm_info}", fontsize=8, ha='right')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    



 



def afficher_schema_transport(p, x, y, mu, nu):
     
    """
    Affiche un schéma du transport entre les distributions
    S'adapte automatiquement à la dimension des discrétisations
    """
    from matplotlib.patches import FancyArrowPatch
    
    # Seuils pour déterminier si on affiche tout ou si on sous-échantillonne
    SEUIL_COMPLET = 20  # En dessous de ce nombre, affichage complet
    
    # Créer la figure
    plt.figure(figsize=(12, 6))
    
    # Dimensions
    m, n = len(x), len(y)
    affichage_complet = m <= SEUIL_COMPLET and n <= SEUIL_COMPLET
    
    # Facteurs de sous-échantillonnage (1 = tous les points, 2 = un point sur deux, etc.)
    if affichage_complet:
        facteur_x = 1
        facteur_y = 1
    else:
        facteur_x = max(1, m // 20)  # Environ 20 points pour x
        facteur_y = max(1, n // 20)  # Environ 20 points pour y
    
    # Sélectionner les indices à afficher
    indices_x = range(0, m, facteur_x)
    indices_y = range(0, n, facteur_y)
    
    # Sous-ensembles des données à afficher
    x_plot = x[indices_x]
    mu_plot = mu[indices_x]
    y_plot = y[indices_y]
    nu_plot = nu[indices_y]
    
    # Échelle pour la taille des points (proportionnelle à la masse)
    scale_mu = 100 / max(mu)
    scale_nu = 100 / max(nu)
    
    # Position des points sur l'axe Y (pour la visualisation)
    pos_mu = np.zeros_like(x_plot)
    pos_nu = np.ones_like(y_plot)
    
    # Tracer les points de la distribution source
    plt.scatter(x_plot, pos_mu, s=mu_plot*scale_mu, c='blue', alpha=0.7, label='Source (μ)')
    
    # Ajouter les annotations pour la source (seulement si affichage complet)
    if affichage_complet:
        for i, (xi, mi) in enumerate(zip(x_plot, mu_plot)):
            plt.annotate(f"x{indices_x[i]}={xi:.2f}\nμ={mi:.2f}", xy=(xi, pos_mu[i]), xytext=(0, -20),textcoords="offset points",ha='center')
    else:
        # Sinon, ajouter quelques annotations sélectionnées
        for i in [0, len(x_plot)//2, len(x_plot)-1]:  # Premier, milieu, dernier
            if i < len(x_plot):
                idx = indices_x[i]
                xi, mi = x_plot[i], mu_plot[i]
                plt.annotate(f"x{idx}={xi:.2f}\nμ={mi:.2f}", 
                            xy=(xi, pos_mu[i]), 
                            xytext=(0, -20),
                            textcoords="offset points",
                            ha='center')
    
    # Tracer les points de la distribution cible
    plt.scatter(y_plot, pos_nu, s=nu_plot*scale_nu, c='red', alpha=0.7, label='Cible (ν)')
    
    # Ajouter les annotations pour la cible
    if affichage_complet:
        for j, (yj, nj) in enumerate(zip(y_plot, nu_plot)):
            plt.annotate(f"y{indices_y[j]}={yj:.2f}\nν={nj:.2f}", xy=(yj, pos_nu[j]), xytext=(0, 10),textcoords="offset points", ha='center')
    else:
        # Sinon, ajouter quelques annotations sélectionnées
        for j in [0, len(y_plot)//2, len(y_plot)-1]:  # Premier, milieu, dernier
            if j < len(y_plot):
                idx = indices_y[j]
                yj, nj = y_plot[j], nu_plot[j]
                plt.annotate(f"y{idx}={yj:.2f}\nν={nj:.2f}", 
                            xy=(yj, pos_nu[j]), 
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha='center')
    
    # Tracer les flèches pour le transport (uniquement pour les plus significatives)
    # Limiter le nombre total de flèches
    max_arrows = 500 if not affichage_complet else 500
    
    # Seuil adaptatif basé sur le maximum de p
    if affichage_complet:
        # Pour l'affichage complet, utiliser un seuil bas
        threshold = 0.01 * np.max(p)
    else:
        # Pour l'affichage échantillonné, ajuster le seuil en fonction du nombre max de flèches
        p_subset = p[np.ix_(indices_x, indices_y)]
        p_flat = p_subset.flatten()
        p_sorted = np.sort(p_flat)
        if len(p_sorted) > max_arrows:
            threshold = p_sorted[-max_arrows]
        else:
            threshold = 0.05 * np.max(p_subset)
    
    # Compteur de flèches
    arrow_count = 0
    
    # Tracer les flèches importantes
    for i_plot, i in enumerate(indices_x):
        for j_plot, j in enumerate(indices_y):
            if p[i, j] > threshold and arrow_count < max_arrows:
                # L'épaisseur de la flèche est proportionnelle à p[i,j]
                width = 0.5 + 3.5 * (p[i, j] / np.max(p))
                
                arrow = FancyArrowPatch(
                    (x[i], pos_mu[i_plot]),
                    (y[j], pos_nu[j_plot]),
                    arrowstyle='-|>',
                    mutation_scale=10,
                    alpha=0.6,
                    linewidth=width,
                    color='gray'
                )
                plt.gca().add_patch(arrow)
                arrow_count += 1
    
    # Ajuster les limites du graphique
    x_min, x_max = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
    margin = 0.1 * (x_max - x_min)
    plt.xlim(x_min - margin, x_max + margin)
    plt.ylim(-0.2, 1.2)
    plt.yticks([])
    
    # Ajouter des infos sur l'échantillonnage dans le titre
    if upper :     
        titre = "Schéma du Transport Optimal de Martingale (Borne sup)"
        
    else :
        titre = "Schéma du Transport Optimal de Martingale (Borne inf)"
    
        
    if not affichage_complet:
        titre += f" (échantillonné: 1/{facteur_x} points source, 1/{facteur_y} points cible, {arrow_count} flèches)"
    plt.title(titre)
    
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    



def afficher_historique_violation(historique):
    
    """
    Affiche l'évolution des violations et de la fonction objectif.
    Pour chaque contrainte, affiche les points correspondant
    aux itérations où cette contrainte a été appliquée, reliés par des lignes.
    """
    # Récupérer les données - utiliser les violations spécifiques à chaque contrainte
    iterations = np.array(range(1, len(historique['violations_max']) + 1))
    
    # Séparer les itérations par type de contrainte
    iter_c1 = [i for i in iterations if (i-1) % 3 == 0]
    iter_c2 = [i for i in iterations if (i-1) % 3 == 1]
    iter_c3 = [i for i in iterations if (i-1) % 3 == 2]
    
    # Extraire les violations pour les itérations correspondantes
    viol_mu = [historique['violations_mu'][i-1] for i in iter_c1]  # Après C1
    viol_nu = [historique['violations_nu'][i-1] for i in iter_c2]  # Après C2
    viol_mart = [historique['violations_mart'][i-1] for i in iter_c3]  # Après C3
    
    # Créer une figure avec deux sous-graphiques
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
    
    # Premier graphique: Violations des contraintes
    # Afficher les points reliés par des lignes
    ax2.plot(iter_c1, viol_mu, label='Violation μ (après C1)', markersize=5, alpha=0.8, linewidth=1.5)
    ax2.plot(iter_c2, viol_nu, label='Violation ν (après C2)', markersize=5, alpha=0.8, linewidth=1.5)
    ax2.plot(iter_c3, viol_mart, label='Violation martingale (après C3)', markersize=5, alpha=0.8, linewidth=1.5)
    
    # Ajouter une ligne pour la tolérance avec une annotation
    ax2.axhline(y=1e-6, color='gray', linestyle='--', alpha=0.7)
    ax2.annotate('Tolérance (1e-6)', xy=(iterations[-1]*0.95, 1e-6), xytext=(iterations[-1]*0.95, 1e-6*3), color='gray', fontsize=9)
    
    # Configurer le premier graphique
    ax2.set_yscale('log')
    ax2.set_xlabel('Itération', fontsize=11)
    ax2.set_ylabel('Violation', fontsize=11)
    
    if upper : 
        ax2.set_title('Évolution des violations par type de contrainte (Borne sup)', fontsize=14, fontweight='bold')
    
    else :
        ax2.set_title('Évolution des violations par type de contrainte (Borne inf)', fontsize=14, fontweight='bold')
        
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend(frameon=True, fontsize=10)
    
    # Ajouter un fond plus agréable
    ax2.set_facecolor('#f8f9fa')
    
    # Deuxième graphique: Fonction objectif
    # Récupérer les données de la fonction objectif
    if 'objectifs' in historique and historique['objectifs']:
        obj_iterations = [item[0] for item in historique['objectifs']]
        obj_values = [item[1] for item in historique['objectifs']]
        
        ax1.plot(obj_iterations, obj_values, 'ko-', label='Prix', markersize=4, linewidth=1.5)
        ax1.set_xlabel('Itération', fontsize=11)
        ax1.set_ylabel('Valeur du prix', fontsize=11)
        ax1.set_title('Évolution du prix', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(frameon=True, fontsize=10)
        ax1.set_facecolor('#f8f9fa')
    else:
        ax1.text(0.5, 0.5, "Pas de données disponibles pour la fonction objectif", ha='center', va='center', transform=ax2.transAxes)
    
    # Ajuster les marges et l'espacement
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  
    




def exemple_uniforme_exponentielle(m, n):
    
    a1, b1 = 0, 1
    a2, b2 = 1e-4, 2
    
    mu = lambda x: 1 / (b1 - a1) if a1 <= x <= b1 else 0
    x, mu = discretize_measure(mu, a1, b1, m)
    
    integral_val, _ = quad(exp1, a2, b2)
    nu = lambda x: exp1(x) / integral_val if a2 <= x <= b2 else 0
    y, nu = discretize_measure(nu, a2, b2, n, exp = False)
    
    return mu, nu, x, y



def exemple_uniforme_uniforme(m, n):
    """Exemple simple avec deux lois uniformes"""
    
    a1, b1 = 1, 3
    a2, b2 = 0, 4
    
    mu = lambda x: 1 / (b1 - a1) if a1 <= x <= b1 else 0
    nu = lambda x: 1 / (b2 - a2) if a2 <= x <= b2 else 0
    
    x, mu = discretize_measure(mu, a1, b1, m)
    y, nu = discretize_measure(nu, a2, b2, n)
    
    return mu, nu, x, y



def exemple_gaussiennes_decalees(m, n):
    """
    Exemple avec deux lois gaussiennes de variance différentes.
    """
    # Discrétisation
    
    a1, b1 = 2, 6
    a2, b2 =  0, 8
    
    mu = lambda x: norm.pdf(x,4,1) if a1 <= x <= b1 else 0
    nu = lambda x: norm.pdf(x,4,2) if a2 <= x <= b2 else 0
    
    x, mu = discretize_measure(mu, a1, b1, m)
    y, nu = discretize_measure(nu, a2, b2, n)
    
    return mu, nu, x, y

    
    
def exemple_log_normal(m, n):
    a1, b1 = 1, np.exp(2)
    a2, b2 = 1, np.exp(2)

    μ1 = 0
    σ1 = np.sqrt(0.5)  #espérance et variance des lois sous-jacentes
    
    μ2 = 0
    σ2 = 2
    
    mu = lambda x: lognorm.pdf(x,σ1,np.exp(μ1)) if a1 <= x <= b1 else 0
    nu = lambda x: lognorm.pdf(x,σ2,np.exp(μ2)) if a2 <= x <= b2 else 0
    
    x, mu = discretize_measure(mu, a1, b1, n)
    y, nu = discretize_measure(nu, a2, b2, m)
    
    return mu, nu, x, y



def discretize_measure(mu, a, b, n, exp = False):
    """
    Approximate a measure mu with bounded support [a,b] using a discrete measure mu_n.
    """
    
    if exp :
        very_small_x_vals = np.linspace(a, 0.1, n//3)
        small_x_vals = np.linspace(0.1, 0.5, n//3)
        large_x_vals = np.linspace(0.5, b, n - 2*(n//3) + 2) 
        
        x_vals = np.concatenate([very_small_x_vals, small_x_vals[1:], large_x_vals[1:]])  # Avoiding duplication at 1
        
    else :
        x_vals = np.linspace(a, b, n)


    dx = (b - a) / n   
    mu_n = np.zeros(n)
    
    for i in range(n):
        xi = x_vals[i]
        
        def f1(x): return n * (x - (xi - dx)) * mu(x)
        def f2(x): return n * ((xi + dx) - x) * mu(x)
        
        integral_1, _ = quad(f1, max(a, xi - dx), xi)
        integral_2, _ = quad(f2, xi, min(b, xi + dx))
        
        mu_n[i] = integral_1 + integral_2
    
    # Normalize to ensure sum(mu_n) is approximately 1
    mu_n /= np.sum(mu_n)
    
    return x_vals, mu_n



def plot_transport_plan(p, x_vals, y_vals, title):
    plt.figure()
    plt.imshow(p, cmap='viridis', aspect='auto', origin='lower', extent=[y_vals[0], y_vals[-1], x_vals[0], x_vals[-1]])
    plt.colorbar(label='Probability Mass')
    plt.xlabel('y values')
    plt.ylabel('x values')
    plt.title(title)
    plt.show()




# Programme principal - Exemple d'utilisation
if __name__ == "__main__":
    
#======================================== PARAMÉTRAGE DU PROBLÈME ==========================================================================================

    m = 100
    n = 100
    epsilon = 0.1
    N_iterations = 500    
    tolerance = 1e-6  
    tol = 1e-6   
    
    #On initialise avec q sinon on prend p_0 = 1 / m*n
    init_q = True
    
    #Normalisation lors de l'affichage
    normalisation="globale"   # "colonne" ou "globale"
    
    #Afficher les fonctions xi+ et xi- sur la matrice optimale
    show_xi_functions = False
    
    # Fonction de coût: G(x,y) = |x-y|
    base_cost_func = lambda x, y: np.abs(x-y)  # normal : "lambda x, y: np.abs(x-y)" lookback : "lambda x, y: max(x,y)" asian : "lambda x, y: (x+y)/2"
    
    exemple = "uniforme_uniforme"  # Options: "uniforme_uniforme", "uniforme_exponentielle", "gaussiennes_decalees", "log_normal"
    
#===========================================================================================================================================================
    
    
    
    
    
    
    
    
    
    # Charger l'exemple choisi
    if exemple == "uniforme_uniforme":
        mu, nu, x, y = exemple_uniforme_uniforme(m, n)
        print("Exemple chargé: Deux lois uniformes")
    elif exemple == "uniforme_exponentielle":
        mu, nu, x, y = exemple_uniforme_exponentielle(m, n)
        print("Exemple chargé: X ~ Uniforme[0,1], Y = X*U où U ~ Exp(1)")
    elif exemple == "gaussiennes_decalees":
        mu, nu, x, y = exemple_gaussiennes_decalees(m, n)
    elif exemple == "log_normal" :
        mu, nu, x, y = exemple_log_normal(m,n)
    else:
        raise ValueError(f"Exemple inconnu: {exemple}")
        
    print(f"Dimensions: {len(mu)}x{len(nu)}")
    
    # Afficher les distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(x, mu, width=x[1]-x[0], alpha=0.7)
    plt.title('Distribution source')
    plt.xlabel('x')
    plt.ylabel('Densité')
    
    plt.subplot(1, 2, 2)
    plt.bar(y, nu, width=y[1]-y[0], alpha=0.7)
    plt.title('Distribution cible')
    plt.xlabel('y')
    plt.ylabel('Densité')
    plt.tight_layout()
    plt.show()
    
    
    cost_func = base_cost_func
    upper = True
    
    # Résoudre le problème pour la borne inférieure
    print("\n=== Calcul de la borne inférieure ===")
    p_upper, historique_upper = resoudre_mot(mu, nu, x, y, cost_func, N_iterations, epsilon=epsilon, tol=tol, tolerance=tolerance)
    
    # Calculer la matrice de coût originale
    G_matrix = np.array([[cost_func(xi, yj) for yj in y] for xi in x])
    expected_upper = np.sum(p_upper * G_matrix)
    
    
    #plot_transport_plan(p_upper, x, y, 'Matrice de Transport Optimal (Borne sup)')
    
    afficher_matrice_transport(p_upper, x, y)
    afficher_schema_transport(p_upper, x, y, mu, nu)
    afficher_historique_violation(historique_upper)
    
    
    # Résoudre le problème pour la borne supérieure avec coût négatif
    print("\n=== Calcul de la borne supérieure ===")
    cost_func = lambda x, y: - base_cost_func(x,y)
    upper = False
    
    p_lower, historique_lower = resoudre_mot(mu, nu, x, y, cost_func, N_iterations, epsilon=epsilon, tol=tol, tolerance=tolerance)
    
    G_matrix_lower = np.array([[cost_func(xi, yj) for yj in y] for xi in x])
    expected_lower = - np.sum(p_lower * G_matrix_lower)
    
    # Afficher les bornes
    print("\n=== Résultats des bornes de prix ===")
    print(f"Borne inférieure (minimale): {expected_lower:.6f}")
    print(f"Borne supérieure (maximale): {expected_upper:.6f}")
    
    # (Optional: Keep existing visualization for one of the runs)
    # Afficher les résultats pour la borne inférieure
    #plot_transport_plan(p_lower, x, y, 'Matrice de Transport Optimal (Borne inf)')
    
    afficher_matrice_transport(p_lower, x, y)
    afficher_schema_transport(p_lower, x, y, mu, nu)
    afficher_historique_violation(historique_lower)
    
    plt.show()