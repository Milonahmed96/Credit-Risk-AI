
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_baseline_logistic(X_train, y_train):
    """
    Trains a class-weighted Logistic Regression baseline model.
    """
    log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)
    return log_reg

def train_advanced_xgboost(X_train, y_train):
    """
    Trains a gradient boosting model adjusted for class imbalance.
    """
    class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=class_ratio,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model
