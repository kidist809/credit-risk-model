import pytest
import pandas as pd
from src.data_processing import FeatureEngineer, compute_rfm_and_target


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2', 'C3'],
        'ProductCategory': ['airtime', 'data_bundles', 'airtime', 'movies', 'movies', 'tv'],
        'ChannelId': ['ChannelId_2', 'ChannelId_3', 'ChannelId_2', 'ChannelId_3', 'ChannelId_5', 'ChannelId_2'],
        'Amount': [100.0, 200.0, 150.0, 50.0, 75.0, 300.0],
        'TransactionStartTime': pd.to_datetime([
            '2018-11-15 10:00:00', '2018-11-16 12:00:00', '2018-11-17 08:00:00',
            '2018-10-01 09:00:00', '2018-10-02 14:00:00', '2018-09-01 11:00:00'
        ])
    })


def test_feature_engineer_output_columns(sample_df):
    engineer = FeatureEngineer()
    result = engineer.fit_transform(sample_df)
    assert 'CustomerId' in result.columns
    assert 'total_amount' in result.columns
    assert 'avg_amount' in result.columns
    assert 'transaction_count' in result.columns
    assert 'std_amount' in result.columns


def test_feature_engineer_row_count(sample_df):
    engineer = FeatureEngineer()
    result = engineer.fit_transform(sample_df)
    assert len(result) == sample_df['CustomerId'].nunique()


def test_feature_engineer_aggregations(sample_df):
    engineer = FeatureEngineer()
    result = engineer.fit_transform(sample_df)
    c1 = result[result['CustomerId'] == 'C1'].iloc[0]
    assert c1['total_amount'] == pytest.approx(450.0)
    assert c1['transaction_count'] == 3
    assert c1['avg_amount'] == pytest.approx(150.0)


def test_compute_rfm_target_columns(sample_df):
    result = compute_rfm_and_target(sample_df)
    assert 'CustomerId' in result.columns
    assert 'is_high_risk' in result.columns


def test_compute_rfm_target_binary(sample_df):
    result = compute_rfm_and_target(sample_df)
    assert set(result['is_high_risk'].unique()).issubset({0, 1})


def test_compute_rfm_target_row_count(sample_df):
    result = compute_rfm_and_target(sample_df)
    assert len(result) == sample_df['CustomerId'].nunique()
