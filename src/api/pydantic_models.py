from pydantic import BaseModel, Field
from typing import List


class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction"""
    total_amount: float = Field(..., description="Total transaction amount")
    avg_amount: float = Field(..., description="Average transaction amount")
    std_amount: float = Field(..., description="Standard deviation of transaction amounts")
    transaction_count: int = Field(..., description="Number of transactions")
    avg_hour: float = Field(..., description="Average transaction hour")
    avg_day: float = Field(..., description="Average transaction day")
    avg_month: float = Field(..., description="Average transaction month")
    avg_year: float = Field(..., description="Average transaction year")
    mode_product_category_data_bundles: int = Field(0, description="Product category: data_bundles")
    mode_product_category_financial_services: int = Field(0, description="Product category: financial_services")
    mode_product_category_movies: int = Field(0, description="Product category: movies")
    mode_product_category_other: int = Field(0, description="Product category: other")
    mode_product_category_ticket: int = Field(0, description="Product category: ticket")
    mode_product_category_transport: int = Field(0, description="Product category: transport")
    mode_product_category_tv: int = Field(0, description="Product category: tv")
    mode_product_category_utility_bill: int = Field(0, description="Product category: utility_bill")
    mode_channel_id_ChannelId_2: int = Field(0, description="Channel: ChannelId_2")
    mode_channel_id_ChannelId_3: int = Field(0, description="Channel: ChannelId_3")
    mode_channel_id_ChannelId_5: int = Field(0, description="Channel: ChannelId_5")

    class Config:
        json_schema_extra = {
            "example": {
                "total_amount": 15000.0,
                "avg_amount": 500.0,
                "std_amount": 150.0,
                "transaction_count": 30,
                "avg_hour": 14.5,
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
        }


class PredictionRequest(BaseModel):
    """Request body for batch predictions"""
    customers: List[CustomerFeatures]


class PredictionResponse(BaseModel):
    """Response with risk probability for each customer"""
    predictions: List[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"customer_index": 0, "risk_probability": 0.15, "risk_category": "low"},
                    {"customer_index": 1, "risk_probability": 0.85, "risk_category": "high"}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
