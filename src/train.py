import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from src.preprocessing import get_preprocessor

def train_baseline_pipeline(X_train, y_train, save_path='baseline_pipeline.pkl'):
    """
    Builds and trains the Logistic Regression CV pipeline.
    """
    preprocessor, _, _ = get_preprocessor(X_train)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegressionCV(
            class_weight='balanced', cv=5, scoring='roc_auc', random_state=42, max_iter=1000
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, save_path)
    return pipeline

def train_xgboost_pipeline(X_train, y_train, save_path='xgboost_pipeline.pkl'):
    """
    Builds and trains the XGBoost pipeline adjusted for class imbalance.
    """
    preprocessor, _, _ = get_preprocessor(X_train)
    class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            scale_pos_weight=class_ratio, n_estimators=100, max_depth=4, 
            learning_rate=0.1, subsample=0.8, random_state=42, eval_metric='logloss'
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, save_path)
    return pipeline
