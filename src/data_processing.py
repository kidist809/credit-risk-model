import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Aggregates per customer and one-hot encodes categorical mode columns.
    Returns only numeric features.
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
            mode_product_category=(
                'ProductCategory',
                lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
            ),
            mode_channel_id=(
                'ChannelId',
                lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
            )
        ).reset_index()

        time_agg = df.groupby('CustomerId').agg(
            avg_hour=('hour', 'mean'),
            avg_day=('day', 'mean'),
            avg_month=('month', 'mean'),
            avg_year=('year', 'mean')
        ).reset_index()

        profile = num_agg.merge(mode_agg, on='CustomerId').merge(time_agg, on='CustomerId')
        profile = pd.get_dummies(
            profile,
            columns=['mode_product_category', 'mode_channel_id'],
            drop_first=True
        )
        return profile


def compute_rfm_and_target(df_raw, snapshot_date=None, high_risk_percentile=20):
    """
    Returns DataFrame with CustomerId, is_high_risk.
    Uses RFM scoring: high recency, low frequency, low monetary = high risk.
    Top high_risk_percentile% of customers by composite risk score are labeled 1.
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

    r_min, r_max = rfm['recency'].min(), rfm['recency'].max()
    f_min, f_max = rfm['frequency'].min(), rfm['frequency'].max()
    m_min, m_max = rfm['monetary'].min(), rfm['monetary'].max()

    rfm['r_score'] = (rfm['recency'] - r_min) / (r_max - r_min + 1e-10)
    rfm['f_score'] = 1 - (rfm['frequency'] - f_min) / (f_max - f_min + 1e-10)
    rfm['m_score'] = 1 - (rfm['monetary'] - m_min) / (m_max - m_min + 1e-10)
    rfm['risk_score'] = (rfm['r_score'] + rfm['f_score'] + rfm['m_score']) / 3

    threshold = rfm['risk_score'].quantile(1 - high_risk_percentile / 100)
    rfm['is_high_risk'] = (rfm['risk_score'] >= threshold).astype(int)

    print("\nRFM Target Distribution:")
    print(rfm['is_high_risk'].value_counts())
    print(f"High-risk rate: {rfm['is_high_risk'].mean():.2%}")

    return rfm[['CustomerId', 'is_high_risk']]


def load_raw_data(path='data/raw/data.csv'):
    return pd.read_csv(path)


def build_preprocessing_pipeline():
    """
    Pipeline for final imputation and scaling.
    Feature engineering is done separately to preserve column names.
    """
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def process_and_save(raw_path='data/raw/data.csv',
                     output_features='data/processed/features.csv',
                     output_target='data/processed/target.csv',
                     output_final='data/processed/model_data.csv'):
    df = load_raw_data(raw_path)

    target_df = compute_rfm_and_target(df)
    target_df.to_csv(output_target, index=False)
    print(f"Target saved to {output_target}")

    engineer = FeatureEngineer()
    profile = engineer.fit_transform(df)

    customer_ids = profile['CustomerId']
    X = profile.drop(columns=['CustomerId'])
    feature_columns = X.columns.tolist()

    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)

    X_final = pd.DataFrame(X_processed, columns=feature_columns)
    X_final.insert(0, 'CustomerId', customer_ids)
    X_final.to_csv(output_features, index=False)
    print(f"Features saved to {output_features}")

    final_df = X_final.merge(target_df, on='CustomerId', how='inner')
    final_df.to_csv(output_final, index=False)
    print(f"Final model dataset saved to {output_final}")
    return final_df


if __name__ == '__main__':
    process_and_save()
