from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(X_train):
    """
    Returns a leakage-safe ColumnTransformer for imputation, scaling, and encoding.
    """
    categorical_cols = ['sex', 'education', 'marriage']
    continuous_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), continuous_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), categorical_cols)
    ])
    
    return preprocessor, categorical_cols, continuous_cols
