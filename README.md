# Credit Risk Model for Bati Bank – Buy‑Now‑Pay‑Later Service

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/credit-risk-model/actions/workflows/ci.yml)

## Project Overview

An end-to-end machine learning solution for credit risk scoring using alternative eCommerce transaction data. This project builds a predictive model that assigns risk probabilities to customers applying for Buy-Now-Pay-Later (BNPL) services at Bati Bank.

### Business Context

Bati Bank is partnering with an eCommerce platform to offer BNPL services. Since traditional credit history is unavailable, this model leverages behavioral transaction patterns (RFM analysis) to predict credit risk.

### Key Features

- **Proxy Target Engineering**: RFM-based customer segmentation to define high-risk customers
- **Feature Engineering Pipeline**: Automated transformation from raw transactions to model-ready features
- **Model Comparison**: LogisticRegression, RandomForest, and GradientBoosting with hyperparameter tuning
- **MLflow Tracking**: Complete experiment tracking and model registry
- **REST API**: FastAPI-based prediction service
- **Docker Deployment**: Containerized API for easy deployment
- **CI/CD Pipeline**: Automated testing and linting with GitHub Actions

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well‑documented model?

Basel II requires financial institutions to quantify credit risk using robust, transparent, and defensible methods. Under the Internal Ratings‑Based (IRB) approach, a bank must demonstrate that its risk estimates (Probability of Default, Loss Given Default) are derived from a well‑documented, statistically sound model. Interpretability is critical because:

- **Regulatory audit** – Supervisors must be able to trace how a score is calculated, validate variable selection, and ensure no discriminatory factors are used.
- **Model risk management** – Opaque "black‑box" models increase the risk of undetected biases or errors that could lead to capital misallocation.
- **Business decisions** – Loan officers and credit committees need to explain decisions to customers and internal stakeholders.

Thus, the model must be accompanied by thorough documentation: variable definitions, transformation logic (e.g., Weight of Evidence), performance metrics, and monitoring plans. Even if we later experiment with complex algorithms, we must provide a clear rationale and maintain an interpretable baseline.

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy‑based prediction introduce?

The raw eCommerce transaction data contains no historical loan performance (e.g., 90‑day past due). To build a supervised model we must **engineer a proxy default label**. We use customers' behavioural patterns (Recency, Frequency, Monetary value) to identify a segment that resembles "high‑risk" — typically disengaged, low‑spend customers who are unlikely to repay a future BNPL loan.

**Business risks introduced by a proxy label:**

- **Labelling error** – A customer flagged as high‑risk by RFM clustering might have excellent creditworthiness in reality. Misclassification can lead to lost revenue (false positives) or unexpected defaults (false negatives).
- **Concept drift** – Behaviour patterns may change over time; the proxy definition might become stale and need recalibration.
- **No ground truth** – Model evaluation metrics (accuracy, AUC) are measured against the proxy, not real defaults. We must communicate that this is a **first‑phase model**, to be validated with actual repayment data as soon as it becomes available.
- **Regulatory scrutiny** – The proxy logic must be clearly justified, as it directly affects who receives credit.

### 3. What are the key trade‑offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high‑performance model (e.g., Gradient Boosting) in a regulated financial context?

| Aspect | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|--------|----------------------------|----------------------------------------|
| **Interpretability** | High – coefficients directly indicate risk direction and magnitude; WoE bins are easily explained. | Low – complex feature interactions, partial dependence plots required for rough interpretation. |
| **Regulatory acceptance** | Preferred – aligns with Basel II requirements, especially with a scorecard format. | Often needs extensive documentation and possibly a simpler "challenger" model alongside it. |
| **Performance** | Adequate for linear relationships; may underperform if strong non‑linearities exist. | Typically higher accuracy, better at capturing non‑linear patterns and interactions. |
| **Explainability tools** | Native – odds ratios, score weights, reason codes. | Post‑hoc (SHAP, LIME) required, which adds complexity and may be contested by regulators. |
| **Stability & maintenance** | Very stable, easy to monitor for drift. | More sensitive to input changes; retraining can alter decisions unpredictably. |

**Our approach:** We train both a logistic regression (as the baseline interpretable model) and a gradient boosting model (for performance comparison). The final recommendation weighs the marginal performance gain against the need for transparency, documentation overhead, and regulatory comfort.

## Project Structure

```
credit-risk-model/
├── .github/workflows/
│   └── ci.yml              # CI/CD pipeline
├── data/
│   ├── raw/                # Raw transaction data (gitignored)
│   └── processed/          # Processed features and targets
├── notebooks/
│   └── eda.ipynb           # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Feature engineering pipeline
│   ├── train.py            # Model training and MLflow tracking
│   └── api/
│       ├── main.py         # FastAPI application
│       └── pydantic_models.py  # Request/response schemas
├── tests/
│   └── test_data_processing.py  # Unit tests
├── mlruns/                 # MLflow experiments (gitignored)
├── mlflow.db               # MLflow tracking database
├── Dockerfile              # Container definition
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── test_api.py             # API testing script
└── README.md
```

## Installation

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Processing and Feature Engineering

```bash
python src/data_processing.py
```

This will:
- Load raw transaction data from `data/raw/data.csv`
- Engineer RFM-based proxy target variable (`is_high_risk`)
- Create aggregated customer features
- Save processed datasets to `data/processed/`

### 2. Model Training

```bash
python src/train.py
```

This will:
- Train LogisticRegression, RandomForest, and GradientBoosting models
- Perform hyperparameter tuning with GridSearchCV
- Log all experiments to MLflow
- Register the best model in MLflow Model Registry

### 3. View MLflow Experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000 in your browser to view:
- Experiment runs with parameters and metrics
- Model comparison charts
- Registered models in the Model Registry

### 4. Run API Locally

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 5. Test API

```bash
python test_api.py
```

### 6. Run Unit Tests

```bash
pytest tests/ -v --cov=src
```

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

API will be available at http://localhost:8000

### Build Docker Image Manually

```bash
docker build -t credit-risk-api:latest .
docker run -p 8000:8000 credit-risk-api:latest
```

## API Usage Examples

### Health Check

```bash
curl http://localhost:8000/
```

### Get Model Info

```bash
curl http://localhost:8000/model-info
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "total_amount": 25000.0,
        "avg_amount": 833.33,
        "std_amount": 250.0,
        "transaction_count": 30,
        "avg_hour": 12.0,
        "avg_day": 15.0,
        "avg_month": 6.0,
        "avg_year": 2018.0,
        "mode_product_category_data_bundles": 1,
        "mode_channel_id_ChannelId_2": 1
      }
    ]
  }'
```

## Model Performance

### Results Summary (from MLflow)

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.9872 | High | High | High |
| Random Forest | 0.9954 | High | High | High |
| **Gradient Boosting** | **0.9965** | **High** | **High** | **High** |

**Selected Model**: GradientBoosting (best ROC-AUC performance)

### Proxy Target Distribution

- Low Risk (0): 80% (2,993 customers)
- High Risk (1): 20% (749 customers)

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Linter

```bash
flake8 src tests --max-line-length=120
```

### Check Code Formatting

```bash
black --check src tests
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:

1. **Lint**: Checks code quality with flake8
2. **Test**: Runs all unit tests with pytest
3. **Build**: Creates Docker image (on main branch)

Workflow triggers on:
- Push to `main` or `task-*` branches
- Pull requests to `main`

## Limitations and Future Work

### Current Limitations

1. **Proxy Target**: The `is_high_risk` label is derived from RFM patterns, not actual default data
2. **No Ground Truth**: Model performance measured against proxy, not real loan outcomes
3. **Concept Drift**: Customer behavior may change over time
4. **Limited Features**: Only transaction-level data available

### Future Enhancements

1. **Validate with Real Data**: Compare proxy labels against actual BNPL repayment data once available
2. **Additional Features**: Incorporate external data (demographics, credit bureau scores)
3. **Model Monitoring**: Implement drift detection and automatic retraining
4. **Scorecard Development**: Convert probability to traditional credit scores (300-850)
5. **Explainability**: Add SHAP values for model interpretability
6. **A/B Testing**: Deploy shadow mode alongside existing credit decisioning

## Screenshots for Submission

### MLflow Tracking UI
1. Navigate to http://localhost:5000 after running `mlflow ui`
2. Screenshot the experiments page showing all 3 model runs
3. Screenshot the model registry showing registered `CreditRiskModel`

### Unit Tests
Run `pytest tests/ -v` and screenshot the passing tests

### API Documentation
Navigate to http://localhost:8000/docs and screenshot the Swagger UI

### Docker Running
Run `docker-compose up` and screenshot the running container

### CI/CD Pipeline
Push to GitHub and screenshot the passing workflow in Actions tab

## Contributors

- Your Name
- Team Members

## License

This project is for educational purposes as part of the 10 Academy training program.

## References

- [Basel II Capital Accord](https://www.bis.org/publ/bcbs128.htm)
- [Credit Scoring Approaches Guidelines - World Bank](https://openknowledge.worldbank.org/handle/10986/26098)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
