import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.features import engineer_banking_features

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans hidden undocumented values in categorical features.
    """
    df = df.copy()
    df['education'] = df['education'].replace([0, 5, 6], 4)
    df['marriage'] = df['marriage'].replace(0, 3)
    return df

def preprocess_pipeline(df: pd.DataFrame):
    """
    Executes the full preprocessing pipeline: cleaning, feature engineering, 
    encoding, splitting, and scaling.
    """
    # Clean and Engineer
    df = clean_data(df)
    df = engineer_banking_features(df)
    
    # Define Column Groups
    categorical_cols = ['sex', 'education', 'marriage']
    pay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    engineered_features = ['utilization_ratio', 'pay_to_bill_ratio', 'avg_payment_delay', 'bill_trend']
    continuous_cols = ['limit_bal', 'age'] + \
                      [f'bill_amt{i}' for i in range(1, 7)] + \
                      [f'pay_amt{i}' for i in range(1, 7)] + \
                      engineered_features
                      
    # One-Hot Encoding
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_processed.drop('default', axis=1)
    y = df_processed['default']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
