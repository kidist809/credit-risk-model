from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.sklearn
import pandas as pd
import logging

from .pydantic_models import PredictionRequest, PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="REST API for credit risk scoring using alternative eCommerce data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_version = None

MLFLOW_TRACKING_URI = "sqlite:///C:/Users/kidim/credit-risk-model/mlflow.db"
MODEL_NAME = "CreditRiskModel"

EXPECTED_FEATURES = [
    'total_amount', 'avg_amount', 'std_amount', 'transaction_count',
    'avg_hour', 'avg_day', 'avg_month', 'avg_year',
    'mode_product_category_data_bundles', 'mode_product_category_financial_services',
    'mode_product_category_movies', 'mode_product_category_other',
    'mode_product_category_ticket', 'mode_product_category_transport',
    'mode_product_category_tv', 'mode_product_category_utility_bill',
    'mode_channel_id_ChannelId_2', 'mode_channel_id_ChannelId_3',
    'mode_channel_id_ChannelId_5'
]


@app.on_event("startup")
async def load_model():
    global model, model_version
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        model_version = latest_versions[0].version if latest_versions else "unknown"

        logger.info(f"Model loaded: {MODEL_NAME} version {model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version or "unknown"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk probability for one or more customers.
    Returns risk_probability (0-1) and risk_category (low/medium/high).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        customers_data = [customer.dict() for customer in request.customers]
        df = pd.DataFrame(customers_data)

        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[EXPECTED_FEATURES]

        probabilities = model.predict_proba(df)[:, 1]

        predictions = []
        for idx, prob in enumerate(probabilities):
            risk_category = "low" if prob < 0.3 else ("medium" if prob < 0.7 else "high")
            predictions.append({
                "customer_index": idx,
                "risk_probability": float(prob),
                "risk_category": risk_category
            })

        logger.info(f"Predicted {len(predictions)} customers")
        return PredictionResponse(predictions=predictions)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
