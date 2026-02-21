import pandas as pd
import numpy as np

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans hidden categorical values and engineers banking features.
    """
    df = df.copy()
    
    # 1. Clean Data
    df['education'] = df['education'].replace([0, 5, 6], 4)
    df['marriage'] = df['marriage'].replace(0, 3)
    
    # 2. Engineer Features
    df['utilization_ratio'] = (df['bill_amt1'] / df['limit_bal']).clip(lower=0)
    df['pay_to_bill_ratio'] = np.where(df['bill_amt1'] > 0, df['pay_amt1'] / df['bill_amt1'], 1.0)
    df['pay_to_bill_ratio'] = np.clip(df['pay_to_bill_ratio'], 0, 1)
    
    pay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    df['avg_payment_delay'] = df[pay_cols].clip(lower=0).mean(axis=1)
    df['bill_trend'] = df['bill_amt1'] - df['bill_amt6']
    
    return df
