import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Master transformer that performs:
    - Aggregation (Total, Avg, Std, Count)
    - Mode of categoricals per customer
    - Time feature extraction (hour, day, month, year)
    - One‑hot encoding of categoricals
    - Imputation & scaling handled later in pipeline
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Convert to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # 1. Numeric aggregations per CustomerId
        num_agg = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std'),
            transaction_count=('TransactionId', 'count')
        ).reset_index()

        # 2. Most frequent ProductCategory and ChannelId per customer
        mode_agg = df.groupby('CustomerId').agg(
            mode_product_category=('ProductCategory', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
            mode_channel_id=('ChannelId', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        ).reset_index()

        # 3. Time features: take the latest transaction's time values?
        # For simplicity, use the average transaction hour/day/month/year per customer.
        time_df = df.groupby('CustomerId').agg(
            avg_hour=('hour', 'mean'),
            avg_day=('day', 'mean'),
            avg_month=('month', 'mean'),
            avg_year=('year', 'mean')
        ).reset_index()

        # Merge all aggregations
        customer_profile = num_agg.merge(mode_agg, on='CustomerId').merge(time_df, on='CustomerId')

        # Drop CustomerId for modeling
        customer_profile.drop(columns=['CustomerId'], inplace=True)

        # One‑hot encode categoricals
        cat_cols = ['mode_product_category', 'mode_channel_id']
        customer_profile = pd.get_dummies(customer_profile, columns=cat_cols, drop_first=True)

        return customer_profile

def load_raw_data(path='data/raw/Xente.csv'):
    df = pd.read_csv(path)
    # Create time columns if not already (needed for aggregation)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['day'] = df['TransactionStartTime'].dt.day
    df['month'] = df['TransactionStartTime'].dt.month
    df['year'] = df['TransactionStartTime'].dt.year
    return df

def build_preprocessing_pipeline():
    """
    Returns a Pipeline that:
    - Applies custom feature engineering (FeatureEngineer)
    - Imputes missing values (median for numeric)
    - Standardises numeric features
    """
    pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline

# Optional: a wrapper that fits and transforms and saves processed data
def process_and_save(raw_path='data/raw/Xente.csv', output_path='data/processed/features.csv'):
    df = load_raw_data(raw_path)
    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(df)
    # X_processed is a numpy array; we need column names from the pipeline
    # We'll retrieve feature names from the last step that has them.
    # For simplicity, we'll convert back to DataFrame with generic column names
    # but later we'll save the pipeline to keep consistent.
    pd.DataFrame(X_processed).to_csv(output_path, index=False)
    print(f"Processed features saved to {output_path}")
    return pipeline

if __name__ == '__main__':
    pipeline = process_and_save()