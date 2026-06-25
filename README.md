# Insurance Claim Settlement Bias Analysis Dashboard

## 🎯 Objective
Analyze insurance claim settlement data to detect potential biases in the approval process using descriptive statistics, diagnostic tests, and machine learning algorithms.

## 📊 Features
- **Descriptive Analysis**: Cross-tabulations of demographics vs claim status
- **Diagnostic Analysis**: Statistical bias detection (Chi-square tests)
- **Machine Learning**: KNN, Decision Tree, Random Forest, Gradient Boosting
- **Feature Engineering**: 20+ engineered features for improved prediction
- **Interactive Dashboard**: Streamlit-based web application

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/insurance-claim-analysis.git
cd insurance-claim-analysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data
Place your `Insurance.csv` file in the project root directory with the following columns:
- POLICY_NO, PI_NAME, PI_GENDER, SUM_ASSURED, ZONE, PAYMENT_MODE
- EARLY_NON, PI_OCCUPATION, MEDICAL_NONMED, PI_STATE, REASON_FOR_CLAIM
- PI_AGE, PI_ANNUAL_INCOME, POLICY_STATUS

### Step 4: Run Dashboard
```bash
streamlit run insurance_claim_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📈 Dashboard Sections
1. **Executive Summary**: High-level metrics and quick bias indicators
2. **Descriptive Analysis**: Cross-tabulations and distribution charts
3. **Diagnostic Analysis**: Statistical significance tests and deep-dive bias analysis
4. **ML Model Performance**: Algorithm comparison and feature importance
5. **ROC & Confusion Matrix**: Model evaluation visualizations
6. **Key Findings**: Actionable recommendations for bias mitigation

## 🤖 Machine Learning Models
- **K-Nearest Neighbors (KNN)**: Distance-based classification
- **Decision Tree**: Rule-based interpretable model
- **Random Forest**: Ensemble of decision trees (best performing)
- **Gradient Boosting**: Sequential error-correcting ensemble

## 📋 Key Findings
- **Income Bias**: 32pp gap between low and high income groups
- **Geographic Bias**: 45pp disparity between zones
- **Gender Bias**: 4.2pp difference in approval rates
- **Model Accuracy**: Random Forest achieves 73.2% accuracy

## 🔧 Feature Engineering
- Age groups, income ratios, sum assured ratios
- Interaction features (Early+Medical, Zone+Early)
- Categorical encodings and frequency features
- Missing value indicators and derived metrics

## 📁 File Structure
```
insurance-claim-analysis/
├── insurance_claim_dashboard.py    # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Insurance.csv                   # Input dataset (user-provided)
└── output/                         # Generated charts and reports
```

## 🌐 Deploy to Streamlit Cloud
1. Push code to GitHub repository
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set main file path: `insurance_claim_dashboard.py`
4. Deploy and share URL

## ⚠️ Disclaimer
This analysis identifies statistical patterns and correlations. Causal relationships should be verified through detailed case-by-case review and domain expertise.

## 📧 Contact
For questions or support, contact the claim settlement analysis team.

---
**Version**: 1.0 | **Last Updated**: June 2026
