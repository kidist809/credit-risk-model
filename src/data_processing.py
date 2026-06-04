import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Aggregates per customer: total, avg, std, count of amounts,
    mode of ProductCategory and ChannelId, average time features.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['hour'] = df['TransactionStartTime'].dt.hour
        df['day'] = df['TransactionStartTime'].dt.day
        df['month'] = df['TransactionStartTime'].dt.month
        df['year'] = df['TransactionStartTime'].dt.year

        num_agg = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std'),
            transaction_count=('TransactionId', 'count')
        ).reset_index()

        mode_agg = df.groupby('CustomerId').agg(
            mode_product_category=('ProductCategory', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
            mode_channel_id=('ChannelId', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        ).reset_index()

        time_agg = df.groupby('CustomerId').agg(
            avg_hour=('hour', 'mean'),
            avg_day=('day', 'mean'),
            avg_month=('month', 'mean'),
            avg_year=('year', 'mean')
        ).reset_index()

        profile = num_agg.merge(mode_agg, on='CustomerId').merge(time_agg, on='CustomerId')
        # Keep CustomerId temporarily for merging target
        return profile  # Returns DataFrame with CustomerId

def compute_rfm_and_target(df_raw, snapshot_date=None):
    """
    df_raw: raw transaction DataFrame with TransactionStartTime, CustomerId, Amount
    Returns: DataFrame with CustomerId, recency, frequency, monetary, is_high_risk
    """
    df = df_raw.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        frequency=('TransactionId', 'nunique'),
        monetary=('Amount', 'sum')
    ).reset_index()

    # Scale for clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

    # K-Means with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify high‑risk cluster: lowest average monetary and frequency
    cluster_stats = rfm.groupby('cluster')[['monetary', 'frequency']].mean()
    # The cluster with the smallest sum of normalized scores is the high‑risk
    cluster_stats['score'] = cluster_stats['monetary'] + cluster_stats['frequency']
    high_risk_cluster = cluster_stats['score'].idxmin()
    
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]

def build_preprocessing_pipeline():
    pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline

def process_and_save(raw_path='data/raw/Xente.csv',
                     output_features='data/processed/features.csv',
                     output_target='data/processed/target.csv',
                     output_final='data/processed/model_data.csv'):
    df = load_raw_data(raw_path)
    
    # Compute target
    target_df = compute_rfm_and_target(df)
    target_df.to_csv(output_target, index=False)
    
    # Compute features
    pipeline = build_preprocessing_pipeline()
    X_with_cust = pipeline.named_steps['feature_engineer'].fit_transform(df)
    # Remove CustomerId before imputation/scaling
    customer_ids = X_with_cust['CustomerId']
    X_numeric = X_with_cust.drop(columns=['CustomerId'])
    
    # Apply imputer and scaler manually (since pipeline expects full array)
    # Actually, we can adjust pipeline to work with DataFrame, but easier:
    # We'll build a new pipeline without the FeatureEngineer for final steps
    # Better: we'll modify the pipeline to keep CustomerId aside
    # Here we'll just do it stepwise for clarity
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_numeric)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Convert back to DataFrame with column names
    feature_names = X_numeric.columns.tolist()
    X_processed = pd.DataFrame(X_scaled, columns=feature_names)
    X_processed.insert(0, 'CustomerId', customer_ids)
    
    # Save features
    X_processed.to_csv(output_features, index=False)
    
    # Merge with target
    final_df = X_processed.merge(target_df, on='CustomerId', how='inner')
    final_df.to_csv(output_final, index=False)
    print(f"Final dataset with target saved to {output_final}")
    return final_df

def load_raw_data(path='data/raw/Xente.csv'):
    return pd.read_csv(path)

if __name__ == '__main__':
    process_and_save()