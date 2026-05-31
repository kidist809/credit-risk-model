# Credit Risk Model for Bati Bank – Buy‑Now‑Pay‑Later Service

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence the need for an interpretable and well‑documented model?

Basel II requires financial institutions to quantify credit risk using robust, transparent, and defensible methods. Under the Internal Ratings‑Based (IRB) approach, a bank must demonstrate that its risk estimates (Probability of Default, Loss Given Default) are derived from a well‑documented, statistically sound model. Interpretability is critical because:

- **Regulatory audit** – Supervisors must be able to trace how a score is calculated, validate variable selection, and ensure no discriminatory factors are used.
- **Model risk management** – Opaque “black‑box” models increase the risk of undetected biases or errors that could lead to capital misallocation.
- **Business decisions** – Loan officers and credit committees need to explain decisions to customers and internal stakeholders.

Thus, the model must be accompanied by thorough documentation: variable definitions, transformation logic (e.g., Weight of Evidence), performance metrics, and monitoring plans. Even if we later experiment with complex algorithms, we must provide a clear rationale and maintain an interpretable baseline.

### 2. Without a direct “default” label, why is a proxy variable necessary, and what business risks does proxy‑based prediction introduce?

The raw eCommerce transaction data contains no historical loan performance (e.g., 90‑day past due). To build a supervised model we must **engineer a proxy default label**. We use customers’ behavioural patterns (Recency, Frequency, Monetary value) to identify a segment that resembles “high‑risk” — typically disengaged, low‑spend customers who are unlikely to repay a future BNPL loan.

**Business risks introduced by a proxy label:**

- **Labelling error** – A customer flagged as high‑risk by RFM clustering might have excellent creditworthiness in reality. Misclassification can lead to lost revenue (false positives) or unexpected defaults (false negatives).
- **Concept drift** – Behaviour patterns may change over time; the proxy definition might become stale and need recalibration.
- **No ground truth** – Model evaluation metrics (accuracy, AUC) are measured against the proxy, not real defaults. We must communicate that this is a **first‑phase model**, to be validated with actual repayment data as soon as it becomes available.
- **Regulatory scrutiny** – The proxy logic must be clearly justified, as it directly affects who receives credit.

### 3. What are the key trade‑offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high‑performance model (e.g., Gradient Boosting) in a regulated financial context?

| Aspect | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|--------|----------------------------|----------------------------------------|
| **Interpretability** | High – coefficients directly indicate risk direction and magnitude; WoE bins are easily explained. | Low – complex feature interactions, partial dependence plots required for rough interpretation. |
| **Regulatory acceptance** | Preferred – aligns with Basel II requirements, especially with a scorecard format. | Often needs extensive documentation and possibly a simpler “challenger” model alongside it. |
| **Performance** | Adequate for linear relationships; may underperform if strong non‑linearities exist. | Typically higher accuracy, better at capturing non‑linear patterns and interactions. |
| **Explainability tools** | Native – odds ratios, score weights, reason codes. | Post‑hoc (SHAP, LIME) required, which adds complexity and may be contested by regulators. |
| **Stability & maintenance** | Very stable, easy to monitor for drift. | More sensitive to input changes; retraining can alter decisions unpredictably. |

**Our approach:** We will train both a logistic regression (as the baseline interpretable model) and a gradient boosting model (for performance comparison). The final recommendation will weigh the marginal performance gain against the need for transparency, documentation overhead, and regulatory comfort.
