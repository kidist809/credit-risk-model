import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Health Check Response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_model_info():
    """Test the model info endpoint"""
    response = requests.get(f"{BASE_URL}/model-info")
    print("Model Info Response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction():
    """Test the prediction endpoint"""
    # Sample customer data (low risk profile)
    low_risk_customer = {
        "total_amount": 25000.0,
        "avg_amount": 833.33,
        "std_amount": 250.0,
        "transaction_count": 30,
        "avg_hour": 12.0,
        "avg_day": 15.0,
        "avg_month": 6.0,
        "avg_year": 2018.0,
        "mode_product_category_data_bundles": 1,
        "mode_product_category_financial_services": 0,
        "mode_product_category_movies": 0,
        "mode_product_category_other": 0,
        "mode_product_category_ticket": 0,
        "mode_product_category_transport": 0,
        "mode_product_category_tv": 0,
        "mode_product_category_utility_bill": 0,
        "mode_channel_id_ChannelId_2": 1,
        "mode_channel_id_ChannelId_3": 0,
        "mode_channel_id_ChannelId_5": 0
    }
    
    # Sample customer data (high risk profile)
    high_risk_customer = {
        "total_amount": 500.0,
        "avg_amount": 50.0,
        "std_amount": 25.0,
        "transaction_count": 10,
        "avg_hour": 22.0,
        "avg_day": 28.0,
        "avg_month": 12.0,
        "avg_year": 2017.0,
        "mode_product_category_data_bundles": 0,
        "mode_product_category_financial_services": 0,
        "mode_product_category_movies": 0,
        "mode_product_category_other": 1,
        "mode_product_category_ticket": 0,
        "mode_product_category_transport": 0,
        "mode_product_category_tv": 0,
        "mode_product_category_utility_bill": 0,
        "mode_channel_id_ChannelId_2": 0,
        "mode_channel_id_ChannelId_3": 0,
        "mode_channel_id_ChannelId_5": 1
    }
    
    # Make prediction request
    payload = {
        "customers": [low_risk_customer, high_risk_customer]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Prediction Response:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Bati Bank Credit Risk API")
    print("=" * 60)
    print()
    
    try:
        test_health_check()
        test_model_info()
        test_prediction()
        print("✅ All API tests passed!")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure the server is running.")
        print("Run: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
