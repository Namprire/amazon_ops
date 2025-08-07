import numpy as np
import pandas as pd
from pymcdm.methods import TOPSIS
from pymcdm.helpers import rrankdata
import math

def build_ahp_weights(pairwise_matrix):
    """
    Compute AHP weights using the principal eigenvector method.
    """
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    max_index = np.argmax(np.real(eigvals))
    weights = np.real(eigvecs[:, max_index])
    normalized_weights = weights / np.sum(weights)
    return normalized_weights

def consistency_ratio(pairwise_matrix):
    """
    Calculate AHP consistency ratio (CR).
    CR < 0.1: Good, 0.1-0.2: Borderline, >0.2: Bad
    """
    n = pairwise_matrix.shape[0]
    if n < 3:
        return 0.0
    
    # Random Index values for matrices of size n
    RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    if n not in RI:
        return 0.0
    
    # Calculate maximum eigenvalue
    eigvals = np.linalg.eigvals(pairwise_matrix)
    lambda_max = np.max(np.real(eigvals))
    
    # Consistency Index
    CI = (lambda_max - n) / (n - 1)
    
    # Consistency Ratio
    CR = CI / RI[n]
    return CR

def apply_time_discounting(weights, discount_rate):
    """
    Apply exponential time discounting to criteria weights.
    """
    decay = np.exp(-discount_rate * np.arange(len(weights)))
    discounted = weights * decay
    return discounted / np.sum(discounted)

def rank_with_topsis(matrix, weights, types):
    """
    Rank career options using TOPSIS method.
    """
    model = TOPSIS()
    scores = model(matrix, weights, types)
    ranking = rrankdata(scores)
    return ranking, scores

def load_demo_dataset(path='data/demo.csv'):
    """
    Load demo dataset from CSV file.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        # Return default demo data if file not found
        demo_data = {
            'Income': [9, 7, 5, 3, 4],
            'Working Hours': [60, 45, 40, 30, 55],
            'Stability': [4, 6, 9, 5, 7]
        }
        df = pd.DataFrame(demo_data)
        df.index = [
            "High-Risk High-Reward",
            "Balanced Private Sector", 
            "Stable Public Sector",
            "Flexible Low-Income",
            "Mission-Driven Role"
        ]
        return df

def load_realistic_jobs_dataset():
    """
    Load realistic jobs dataset with actual career options.
    """
    # Based on typical job market data and research
    jobs_data = {
        'Job Title': [
            'Software Engineer',
            'Investment Banker',
            'Teacher (Public)',
            'Nurse (RN)',
            'Government Officer',
            'Freelance Designer',
            'Management Consultant',
            'Non-Profit Manager',
            'Sales Representative',
            'Research Scientist'
        ],
        'Income': [8, 10, 4, 6, 5, 5, 9, 4, 7, 6],  # 1-10 scale
        'Working Hours': [45, 70, 50, 45, 40, 30, 60, 45, 50, 45],  # Hours per week
        'Stability': [7, 3, 9, 8, 9, 3, 5, 6, 5, 7],  # 1-10 scale (10 = very stable)
        'Description': [
            'Tech industry, high growth potential, flexible work',
            'Financial sector, high stress, excellent compensation',
            'Education sector, stable employment, social impact',
            'Healthcare, stable demand, meaningful work',
            'Public sector, excellent job security, steady career',
            'Creative field, flexible schedule, income variability',
            'Business services, travel required, high earning potential',
            'Social sector, mission-driven, moderate compensation',
            'Business development, performance-based, networking opportunities',
            'Academic/Industry research, intellectual stimulation, grant-dependent'
        ]
    }
    
    df = pd.DataFrame(jobs_data)
    df.set_index('Job Title', inplace=True)
    return df

def normalize_data(df, criteria_cols):
    """
    Normalize data to 1-10 scale for consistency.
    """
    df_normalized = df.copy()
    for col in criteria_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            # Normalize to 1-10 scale
            df_normalized[col] = 1 + 9 * (df[col] - min_val) / (max_val - min_val)
    return df_normalized

def analyze_results(scores, labels, decision_matrix, criteria, discounted_weights):
    """
    Generate textual analysis of TOPSIS results.
    """
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)
    best_option = labels[best_idx]
    worst_option = labels[worst_idx]
    
    # Find most important criterion
    top_criterion_idx = np.argmax(discounted_weights)
    top_criterion = criteria[top_criterion_idx]
    
    # Get performance of best option on top criterion
    best_performance = decision_matrix[best_idx, top_criterion_idx]
    
    analysis = {
        'best_option': best_option,
        'worst_option': worst_option,
        'best_score': scores[best_idx],
        'worst_score': scores[worst_idx],
        'top_criterion': top_criterion,
        'top_criterion_weight': discounted_weights[top_criterion_idx],
        'best_performance': best_performance,
        'score_range': scores.max() - scores.min()
    }
    
    return analysis

def radar_chart(ax, values, labels, label_name="", color="blue", alpha=0.25):
    """
    Create a radar chart on given axes.
    """
    N = len(values)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    vals = list(values) + [values[0]]
    
    ax.plot(angles, vals, 'o-', linewidth=2, label=label_name, color=color)
    ax.fill(angles, vals, alpha=alpha, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.1)
    
    return ax