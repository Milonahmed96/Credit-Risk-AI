import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

def print_evaluation_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Prints standard banking evaluation metrics including ROC-AUC and PR-AUC.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    print(f"Metrics at Threshold = {threshold}")
    print(f"ROC-AUC Score: {roc_auc_score(y_true, y_pred_prob):.4f}")
    print(f"PR-AUC Score:  {average_precision_score(y_true, y_pred_prob):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Default (0)', 'Default (1)']))

def optimize_business_threshold(y_true, y_pred_prob, cost_fn=5000, cost_fp=1000):
    """
    Simulates a business cost matrix to find the optimal decision threshold 
    that minimizes expected portfolio loss.
    """
    thresholds = np.linspace(0.1, 0.9, 100)
    costs = []
    
    for t in thresholds:
        y_pred_t = (y_pred_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)
        
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]
    
    return optimal_threshold, min_cost, thresholds, costs
