import pandas as pd
import numpy as np

def engineer_banking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers high-value banking features from raw credit data.
    """
    df = df.copy()
    
    # 1. Utilization Ratio
    df['utilization_ratio'] = (df['bill_amt1'] / df['limit_bal']).clip(lower=0)
    
    # 2. Pay-to-Bill Ratio
    df['pay_to_bill_ratio'] = np.where(
        df['bill_amt1'] > 0, 
        df['pay_amt1'] / df['bill_amt1'], 
        1.0
    )
    df['pay_to_bill_ratio'] = np.clip(df['pay_to_bill_ratio'], 0, 1)
    
    # 3. Average Payment Delay
    pay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    df['avg_payment_delay'] = df[pay_cols].clip(lower=0).mean(axis=1)
    
    # 4. Bill Trend
    df['bill_trend'] = df['bill_amt1'] - df['bill_amt6']
    
    return df
