"""
LoanIQ - Loan Eligibility and Repayment Risk API
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

ELIGIBILITY_PATH = MODEL_DIR / "eligibility_model.pkl"
REPAYMENT_PATH = MODEL_DIR / "repayment_model.pkl"

APPROVAL_DATA = Path.home() / "Downloads" / "loan_approval_dataset.csv"
REPAYMENT_DATA = Path.home() / "Downloads" / "Loan_default.csv"


# ============================================================
# LOAD MODELS
# ============================================================

try:
    eligibility_model = joblib.load(ELIGIBILITY_PATH)
    print("Eligibility model loaded successfully!")
except Exception as e:
    eligibility_model = None
    print("WARNING: Eligibility model could not be loaded.")
    print(e)


try:
    repayment_model = joblib.load(REPAYMENT_PATH)
    print("Repayment model loaded successfully!")
except Exception as e:
    repayment_model = None
    print("WARNING: Repayment model could not be loaded.")
    print(e)


# ============================================================
# SCALE MATCHING
# ============================================================

INCOME_SCALE = 1.0
LOAN_SCALE = 1.0


try:

    approval_data = pd.read_csv(APPROVAL_DATA)
    repayment_data = pd.read_csv(REPAYMENT_DATA)

    # Remove accidental spaces from CSV column names
    approval_data.columns = approval_data.columns.str.strip()
    repayment_data.columns = repayment_data.columns.str.strip()

    approval_income_mean = approval_data["income_annum"].mean()
    approval_loan_mean = approval_data["loan_amount"].mean()

    repayment_income_mean = repayment_data["Income"].mean()
    repayment_loan_mean = repayment_data["LoanAmount"].mean()

    if approval_income_mean != 0:
        INCOME_SCALE = repayment_income_mean / approval_income_mean

    if approval_loan_mean != 0:
        LOAN_SCALE = repayment_loan_mean / approval_loan_mean

    print("Dataset scale matching calculated.")
    print("Income scale:", INCOME_SCALE)
    print("Loan scale:", LOAN_SCALE)

except Exception as e:

    print("WARNING: Dataset scale matching unavailable.")
    print(e)


# ============================================================
# MODEL FEATURE INFORMATION
# ============================================================

ELIGIBILITY_COLUMNS = []
REPAYMENT_COLUMNS = []

ELIGIBILITY_CATEGORIES = {}
REPAYMENT_CATEGORIES = {}

eligibility_preprocessor = None
repayment_preprocessor = None


def get_model_columns(model):

    if model is None:
        return []

    # Pipeline with feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Pipeline whose first step has feature_names_in_
    if hasattr(model, "named_steps"):

        for _, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return []


if eligibility_model is not None:

    ELIGIBILITY_COLUMNS = get_model_columns(
        eligibility_model
    )


if repayment_model is not None:

    REPAYMENT_COLUMNS = get_model_columns(
        repayment_model
    )


# ============================================================
# PREPROCESSOR INFORMATION
# ============================================================

def get_preprocessor(model):

    if model is None:
        return None

    if hasattr(model, "named_steps"):

        if "preprocessing" in model.named_steps:
            return model.named_steps["preprocessing"]

        # Find a transformer automatically
        for _, step in model.named_steps.items():

            if hasattr(step, "transformers_"):
                return step

    return None


eligibility_preprocessor = get_preprocessor(
    eligibility_model
)

repayment_preprocessor = get_preprocessor(
    repayment_model
)


def get_categories(preprocessor):

    categories = {}

    if preprocessor is None:
        return categories

    try:

        encoder = preprocessor.named_transformers_[
            "categorical"
        ]

        categorical_columns = next(
            transformer[2]
            for transformer in preprocessor.transformers_
            if transformer[0] == "categorical"
        )

        categories = dict(
            zip(
                categorical_columns,
                encoder.categories_
            )
        )

    except Exception as e:

        print(
            "Could not read categorical model information:",
            e
        )

    return categories


ELIGIBILITY_CATEGORIES = get_categories(
    eligibility_preprocessor
)

REPAYMENT_CATEGORIES = get_categories(
    repayment_preprocessor
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="LoanIQ",
    description=(
        "Loan Eligibility and Repayment Risk "
        "Decision Support System"
    ),
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# APPROVAL INPUT SCHEMA
# ============================================================

class ApprovalApplicant(BaseModel):

    no_of_dependents: int = Field(
        ge=0,
        le=20
    )

    education: str

    self_employed: str

    income_annum: float = Field(
        gt=0
    )

    loan_amount: float = Field(
        gt=0
    )

    loan_term: float = Field(
        gt=0
    )

    cibil_score: float = Field(
        ge=300,
        le=900
    )

    residential_assets_value: float = Field(
        ge=0
    )

    commercial_assets_value: float = Field(
        ge=0
    )

    luxury_assets_value: float = Field(
        ge=0
    )

    bank_asset_value: float = Field(
        ge=0
    )


# ============================================================
# REPAYMENT RISK INPUT SCHEMA
# ============================================================

class RiskApplicant(BaseModel):

    age: float = Field(
        ge=18,
        le=100
    )

    education: str

    employment_type: str

    months_employed: float = Field(
        ge=0
    )

    marital_status: str

    no_of_dependents: int = Field(
        ge=0,
        le=20
    )

    income: float = Field(
        gt=0
    )

    loan_amount: float = Field(
        gt=0
    )

    credit_score: float = Field(
        ge=300,
        le=900
    )

    dti_ratio: float = Field(
        ge=0,
        le=1
    )

    num_credit_lines: float = Field(
        ge=0
    )

    interest_rate: float = Field(
        ge=0
    )

    loan_term_months: float = Field(
        gt=0
    )

    loan_purpose: str

    has_mortgage: str

    has_cosigner: str

    residential_assets: float = Field(
        ge=0
    )

    commercial_assets: float = Field(
        ge=0
    )

    luxury_assets: float = Field(
        ge=0
    )

    bank_assets: float = Field(
        ge=0
    )


# ============================================================
# APPROVAL FEATURE PREPARATION
# ============================================================

def build_eligibility_features(
    a: ApprovalApplicant
):

    row = {

        "no_of_dependents":
            a.no_of_dependents,

        "education":
            a.education.strip(),

        "self_employed":
            a.self_employed.strip(),

        "income_annum":
            a.income_annum,

        "loan_amount":
            a.loan_amount,

        "loan_term":
            a.loan_term,

        "cibil_score":
            a.cibil_score,

        "residential_assets_value":
            a.residential_assets_value,

        "commercial_assets_value":
            a.commercial_assets_value,

        "luxury_assets_value":
            a.luxury_assets_value,

        "bank_asset_value":
            a.bank_asset_value,
    }

    df = pd.DataFrame([row])

    # If model exposes feature names,
    # make sure the exact expected columns are used.
    if ELIGIBILITY_COLUMNS:

        missing = [
            column
            for column in ELIGIBILITY_COLUMNS
            if column not in df.columns
        ]

        if missing:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Approval model expects missing "
                    f"features: {missing}"
                )
            )

        df = df[ELIGIBILITY_COLUMNS]

    return df


# ============================================================
# REPAYMENT FEATURE PREPARATION
# ============================================================

def build_repayment_features(
    a: RiskApplicant
):

    row = {

        "Age":
            a.age,

        "Income":
            a.income * INCOME_SCALE,

        "LoanAmount":
            a.loan_amount * LOAN_SCALE,

        "CreditScore":
            a.credit_score,

        "MonthsEmployed":
            a.months_employed,

        "NumCreditLines":
            a.num_credit_lines,

        "InterestRate":
            a.interest_rate,

        "LoanTerm":
            a.loan_term_months,

        "DTIRatio":
            a.dti_ratio,

        "Education":
            a.education.strip(),

        "EmploymentType":
            a.employment_type.strip(),

        "MaritalStatus":
            a.marital_status.strip(),

        "HasMortgage":
            a.has_mortgage.strip(),

        "HasDependents":
            (
                "Yes"
                if a.no_of_dependents > 0
                else "No"
            ),

        "LoanPurpose":
            a.loan_purpose.strip(),

        "HasCoSigner":
            a.has_cosigner.strip(),
    }

    # Validate categorical values
    for column, allowed_values in (
        REPAYMENT_CATEGORIES.items()
    ):

        if column in row:

            allowed = [
                str(value)
                for value in allowed_values
            ]

            if str(row[column]) not in allowed:

                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Invalid value for {column}: "
                        f"{row[column]}. "
                        f"Allowed values: {allowed}"
                    )
                )

    df = pd.DataFrame([row])

    if REPAYMENT_COLUMNS:

        missing = [
            column
            for column in REPAYMENT_COLUMNS
            if column not in df.columns
        ]

        if missing:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Repayment model expects missing "
                    f"features: {missing}"
                )
            )

        df = df[REPAYMENT_COLUMNS]

    return df


# ============================================================
# PROBABILITY HELPER
# ============================================================

def get_probability(
    model,
    X
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    try:

        probabilities = model.predict_proba(X)[0]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model prediction failed: "
                f"{str(e)}"
            )
        )

    classes = list(model.classes_)

    # Handle numeric binary classification
    if 1 in classes:

        positive_index = classes.index(1)

    # Handle string classification such as
    # Approved / Rejected
    elif "Approved" in classes:

        positive_index = classes.index("Approved")

    elif "approved" in classes:

        positive_index = classes.index("approved")

    # If model uses True / False
    elif True in classes:

        positive_index = classes.index(True)

    else:

        # Last class as fallback
        positive_index = len(probabilities) - 1

    return float(
        probabilities[positive_index]
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "application": "LoanIQ",

        "eligibility_model_loaded":
            eligibility_model is not None,

        "repayment_model_loaded":
            repayment_model is not None,

        "eligibility_features":
            ELIGIBILITY_COLUMNS,

        "repayment_features":
            REPAYMENT_COLUMNS
    }


# ============================================================
# LOAN APPROVAL
# ============================================================

@app.post("/predict-approval")
def predict_approval(
    a: ApprovalApplicant
):

    if eligibility_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Eligibility model is not loaded. "
                "Check models/eligibility_model.pkl"
            )
        )

    try:

        X = build_eligibility_features(a)

        probability = get_probability(
            eligibility_model,
            X
        )

        approved = probability >= 0.5

        return {

            "approved":
                bool(approved),

            "approval_probability":
                round(
                    probability * 100,
                    2
                ),

            "decision":
                (
                    "Approved"
                    if approved
                    else "Rejected"
                )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Approval prediction failed: "
                f"{str(e)}"
            )
        )


# ============================================================
# REPAYMENT RISK
# ============================================================

@app.post("/predict-risk")
def predict_risk(
    a: RiskApplicant
):

    if repayment_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Repayment model is not loaded. "
                "Check models/repayment_model.pkl"
            )
        )

    try:

        X = build_repayment_features(a)

        probability = get_probability(
            repayment_model,
            X
        )

        if probability < 0.35:

            risk_level = "Low"

        elif probability < 0.60:

            risk_level = "Medium"

        else:

            risk_level = "High"

        return {

            "risk_level":
                risk_level,

            "default_probability":
                round(
                    probability * 100,
                    2
                ),

            "assessment":
                (
                    "Low repayment risk"
                    if risk_level == "Low"

                    else
                    "Moderate repayment risk"
                    if risk_level == "Medium"

                    else
                    "High repayment risk"
                )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk prediction failed: "
                f"{str(e)}"
            )
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@app.get("/feature-importance")
def feature_importance():

    if eligibility_model is None:

        raise HTTPException(
            status_code=503,
            detail="Eligibility model is not loaded."
        )

    try:

        if not hasattr(
            eligibility_model,
            "named_steps"
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    "Feature importance is unavailable "
                    "for this model structure."
                )
            )

        if "model" not in (
            eligibility_model.named_steps
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    "Feature importance is unavailable."
                )
            )

        model = eligibility_model.named_steps[
            "model"
        ]

        if not hasattr(
            model,
            "feature_importances_"
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    "Feature importance is unavailable "
                    "for this model."
                )
            )

        importances = model.feature_importances_

        if eligibility_preprocessor is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Preprocessor information is unavailable."
                )
            )

        transformed_names = (
            eligibility_preprocessor
            .get_feature_names_out()
        )

        aggregated = {}

        for name, importance in zip(
            transformed_names,
            importances
        ):

            clean_name = name.split(
                "__",
                1
            )[-1]

            original_name = clean_name

            # Group one-hot encoded categories
            for feature in ELIGIBILITY_COLUMNS:

                if clean_name.startswith(
                    feature + "_"
                ):

                    original_name = feature
                    break

            aggregated[original_name] = (
                aggregated.get(
                    original_name,
                    0
                )
                + float(importance)
            )

        labels = {

            "cibil_score":
                "Credit Score",

            "loan_amount":
                "Loan Amount",

            "income_annum":
                "Annual Income",

            "loan_term":
                "Loan Term",

            "no_of_dependents":
                "Dependents",

            "education":
                "Education",

            "self_employed":
                "Employment",

            "residential_assets_value":
                "Residential Assets",

            "commercial_assets_value":
                "Commercial Assets",

            "luxury_assets_value":
                "Luxury Assets",

            "bank_asset_value":
                "Bank Assets",
        }

        total = sum(
            aggregated.values()
        )

        if total == 0:
            total = 1

        top_features = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True
        )[:6]

        return {

            "features": [

                {
                    "name":
                        labels.get(
                            name,
                            name
                        ),

                    "importance":
                        round(
                            (
                                value / total
                            ) * 100,
                            2
                        )
                }

                for name, value
                in top_features
            ]
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Feature importance failed: "
                f"{str(e)}"
            )
        )


# ============================================================
# SERVE FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=BASE_DIR,
        html=True
    ),
    name="frontend"
)