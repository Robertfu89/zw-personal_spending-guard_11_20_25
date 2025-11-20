import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import torch
import torch.nn as nn
import joblib

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


# ============================================================
# Shared constants / paths
# ============================================================

# For Smart Planner (MLP)
INCOME_COL = "total_income"
TARGET_COL = "cashflow"

FLEX_CATEGORIES = [
    "utilities",
    "groceries",
    "transport",
    "dining_out",
    "entertainment",
    "clothing",
    "other_variable",
    "unexpected_spend",
]

FIXED_CATEGORIES = [
    "housing",
    "insurance",
    "childcare",
    "healthcare",
    "education",
    "debt_min_payments",
    "savings_contrib",
    "investment_contrib",
]

# For Trend Dashboard (LSTM)
DATA_FILE_DEFAULT = "family_spending_trends_500families_24months_with_risk.csv"
MODEL_FILE = "lstm_cashflow_model.pt"
SCALER_FILE = "scaler.pkl"

CATEGORIES_TREND = [
    "housing", "utilities", "groceries", "transport", "insurance",
    "childcare", "healthcare", "entertainment", "dining_out",
    "clothing", "education", "debt_min_payments", "savings_contrib",
    "investment_contrib", "unexpected_spend", "other_variable",
]


# ============================================================
# Helpers: styling, badges
# ============================================================

def status_badge(label: str, color: str = "blue"):
    st.markdown(
        f"<span style='background-color:{color};"
        f"color:white;padding:3px 8px;border-radius:10px;font-size:0.85rem;'>"
        f"{label}</span>",
        unsafe_allow_html=True,
    )


# ============================================================
# PART 1 – Smart Monthly Planner (MLP-based optimizer)
# ============================================================

@dataclass
class ModelArtifacts:
    model: MLPRegressor
    feature_cols: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(n=1000, random_state=42) -> pd.DataFrame:
    """
    Synthetic training data for the MLP model.
    Cached so we don't regenerate it on every rerun.
    """
    np.random.seed(random_state)

    income = np.random.uniform(4717, 14150, size=n)
    family_size = np.random.randint(1, 6, size=n)
    num_earners = np.random.randint(1, 3, size=n)
    age_main = np.random.randint(25, 65, size=n)
    home_owner = np.random.binomial(1, 0.6, size=n)

    def clip_ratio(r):
        return np.clip(r, 0, 0.6)

    housing_ratio = np.where(
        home_owner == 1,
        np.random.normal(0.23, 0.04, size=n),
        np.random.normal(0.30, 0.05, size=n),
    )
    utilities_ratio = np.random.normal(0.07, 0.015, size=n)
    groceries_ratio = np.random.normal(0.12, 0.03, size=n)
    transport_ratio = np.random.normal(0.10, 0.03, size=n)
    insurance_ratio = np.random.normal(0.06, 0.015, size=n)
    childcare_ratio = np.where(
        family_size > 2,
        np.random.normal(0.07, 0.03, size=n),
        np.random.normal(0.02, 0.01, size=n),
    )
    healthcare_ratio = np.random.normal(0.06, 0.02, size=n)
    entertainment_ratio = np.random.normal(0.05, 0.02, size=n)
    dining_ratio = np.random.normal(0.05, 0.02, size=n)
    clothing_ratio = np.random.normal(0.03, 0.01, size=n)
    education_ratio = np.random.normal(0.04, 0.02, size=n)
    debt_min_ratio = np.random.normal(0.07, 0.03, size=n)
    savings_ratio = np.random.normal(0.05, 0.02, size=n)
    invest_ratio = np.random.normal(0.05, 0.03, size=n)
    unexpected_ratio = np.abs(np.random.normal(0.03, 0.03, size=n))
    other_var_ratio = np.random.normal(0.04, 0.02, size=n)

    ratios = [
        housing_ratio, utilities_ratio, groceries_ratio, transport_ratio,
        insurance_ratio, childcare_ratio, healthcare_ratio, entertainment_ratio,
        dining_ratio, clothing_ratio, education_ratio, debt_min_ratio,
        savings_ratio, invest_ratio, unexpected_ratio, other_var_ratio
    ]
    ratios = [clip_ratio(r) for r in ratios]

    (
        housing_ratio, utilities_ratio, groceries_ratio, transport_ratio,
        insurance_ratio, childcare_ratio, healthcare_ratio, entertainment_ratio,
        dining_ratio, clothing_ratio, education_ratio, debt_min_ratio,
        savings_ratio, invest_ratio, unexpected_ratio, other_var_ratio
    ) = ratios

    housing = income * housing_ratio
    utilities = income * utilities_ratio
    groceries = income * groceries_ratio
    transport = income * transport_ratio
    insurance = income * insurance_ratio
    childcare = income * childcare_ratio
    healthcare = income * healthcare_ratio
    entertainment = income * entertainment_ratio
    dining_out = income * dining_ratio
    clothing = income * clothing_ratio
    education = income * education_ratio
    debt_min_payments = income * debt_min_ratio
    savings_contrib = income * savings_ratio
    investment_contrib = income * invest_ratio
    unexpected_spend = income * unexpected_ratio
    other_variable = income * other_var_ratio

    total_outflows = (
        housing + utilities + groceries + transport + insurance + childcare +
        healthcare + entertainment + dining_out + clothing + education +
        debt_min_payments + savings_contrib + investment_contrib +
        unexpected_spend + other_variable
    )

    cashflow = income - total_outflows

    df = pd.DataFrame({
        INCOME_COL: income,
        "family_size": family_size,
        "num_earners": num_earners,
        "age_main": age_main,
        "home_owner": home_owner,
        "housing": housing,
        "utilities": utilities,
        "groceries": groceries,
        "transport": transport,
        "insurance": insurance,
        "childcare": childcare,
        "healthcare": healthcare,
        "entertainment": entertainment,
        "dining_out": dining_out,
        "clothing": clothing,
        "education": education,
        "debt_min_payments": debt_min_payments,
        "savings_contrib": savings_contrib,
        "investment_contrib": investment_contrib,
        "unexpected_spend": unexpected_spend,
        "other_variable": other_variable,
        TARGET_COL: cashflow,
    })

    return df


def train_cashflow_model(df, target_col=TARGET_COL) -> ModelArtifacts:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # You can reduce max_iter if you still want faster initial startup
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return ModelArtifacts(
        model, feature_cols, X_train, X_test, y_train, y_test
    )


@st.cache_resource(show_spinner=True)
def get_trained_mlp_model() -> ModelArtifacts:
    """
    Cached MLP model training.
    Runs once per server start; then reused for all users/sessions.
    """
    df = generate_synthetic_dataset()
    return train_cashflow_model(df)


def optimize_family(row, artifacts, max_cut=0.2, n_steps=5):
    model = artifacts.model
    feature_cols = artifacts.feature_cols

    heavy_flex = ["dining_out", "entertainment", "other_variable", "unexpected_spend"]
    soft_flex = ["utilities", "groceries", "transport", "clothing"]

    income = row[INCOME_COL]
    target_min_cf = 0.05 * income  # 5% of income target

    base = row[feature_cols].copy()
    base_pred = float(model.predict(base.to_frame().T)[0])

    best = base.copy()
    best_pred = base_pred
    best_cut = 0.0
    feasible = False

    for alpha in np.linspace(0, max_cut, n_steps):
        cand = base.copy()

        for c in soft_flex:
            cand[c] = cand[c] * (1 - 0.5 * alpha)
        for c in heavy_flex:
            cand[c] = cand[c] * (1 - alpha)

        pred = float(model.predict(cand.to_frame().T)[0])

        if pred > 0:
            feasible = True
            best = cand
            best_pred = pred
            best_cut = alpha
            break

        if pred > best_pred:
            best = cand
            best_pred = pred
            best_cut = alpha

    meets_5pct = feasible and (best_pred >= target_min_cf)

    return {
        "income": income,
        "original_features": base,
        "optimized_features": best,
        "pred_cashflow_before": base_pred,
        "pred_cashflow_after": best_pred,
        "meets_5pct_constraint": meets_5pct,
        "cut_fraction": best_cut,
        "feasible": feasible,
    }


def log_infeasible_case(mode_label, income, cf_before, best_pred, cut_frac):
    if "infeasible_count" not in st.session_state:
        st.session_state["infeasible_count"] = 0
    st.session_state["infeasible_count"] += 1

    log_path = "infeasible_log.csv"
    row = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "mode": mode_label,
        "income": income,
        "cashflow_original": cf_before,
        "best_pred_cashflow": best_pred,
        "max_cut_used": cut_frac,
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(log_path):
        df_new.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(log_path, index=False)


def smart_planner_page(mode: str):
    st.header("💡 Monthly Smart Spending Planner (MLP Optimizer)")

    if mode == "Demo Mode":
        st.markdown(
            """
            **Demo Mode – Scenario Tuning**

            Use synthetic data + manual inputs to explore how AI can:
            - Ensure **positive cashflow**
            - Target **≥ 5% of income** as surplus (when feasible)
            - Adjust only **flex categories** (utilities, groceries, dining out, etc.)
            """
        )
    else:
        st.markdown(
            """
            **User Mode – Your Own Monthly Plan**

            Enter your real monthly income and spending to:
            - See how much you can safely cut without hurting lifestyle too much  
            - Aim for **positive cashflow** and **≥ 5% of income** surplus  
            - Focus cuts on **flex categories** first  
            """
        )

    mode_label = st.radio(
        "Optimization Intensity:",
        ["Gentle – up to 20% cuts", "Aggressive – up to 40% cuts"],
        key="planner_mode",
    )

    if "Gentle" in mode_label:
        max_cut = 0.20
        n_steps = 5
    else:
        max_cut = 0.40
        n_steps = 9

    artifacts = get_trained_mlp_model()

    st.subheader("1. Household Profile & Spending")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Total Monthly Income ($)", min_value=1000.0, value=6000.0)
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)
        num_earners = st.number_input("Number of Earners", min_value=1, max_value=4, value=2)
        age_main = st.number_input("Age of Main Earner", min_value=18, max_value=80, value=40)
        home_owner = st.selectbox(
            "Home Ownership", [0, 1],
            format_func=lambda x: "Renter" if x == 0 else "Owner"
        )

    with col2:
        st.markdown("### Essential Spending")
        housing = st.number_input("Housing (rent/mortgage)", min_value=0.0, value=1800.0)
        utilities = st.number_input("Utilities", min_value=0.0, value=300.0)
        groceries = st.number_input("Groceries", min_value=0.0, value=800.0)
        transport = st.number_input("Transport", min_value=0.0, value=400.0)
        insurance = st.number_input("Insurance", min_value=0.0, value=300.0)
        childcare = st.number_input("Childcare", min_value=0.0, value=300.0)
        healthcare = st.number_input("Healthcare", min_value=0.0, value=200.0)
        debt_min = st.number_input("Debt Minimum Payments", min_value=0.0, value=400.0)

    st.markdown("### Flexible & Financial Spending")
    col3, col4 = st.columns(2)

    with col3:
        entertainment = st.number_input("Entertainment", min_value=0.0, value=250.0)
        dining_out = st.number_input("Dining Out", min_value=0.0, value=350.0)
        clothing = st.number_input("Clothing", min_value=0.0, value=150.0)

    with col4:
        education = st.number_input("Education", min_value=0.0, value=200.0)
        savings = st.number_input("Savings Contributions", min_value=0.0, value=200.0)
        investment = st.number_input("Investment Contributions", min_value=0.0, value=200.0)
        unexpected = st.number_input("Unexpected Cushion", min_value=0.0, value=250.0)
        other_variable = st.number_input("Misc / Impulse", min_value=0.0, value=200.0)

    if st.button("Run AI Smart Plan"):
        row = {
            INCOME_COL: income,
            "family_size": family_size,
            "num_earners": num_earners,
            "age_main": age_main,
            "home_owner": home_owner,
            "housing": housing,
            "utilities": utilities,
            "groceries": groceries,
            "transport": transport,
            "insurance": insurance,
            "childcare": childcare,
            "healthcare": healthcare,
            "entertainment": entertainment,
            "dining_out": dining_out,
            "clothing": clothing,
            "education": education,
            "debt_min_payments": debt_min,
            "savings_contrib": savings,
            "investment_contrib": investment,
            "unexpected_spend": unexpected,
            "other_variable": other_variable,
        }

        r = pd.DataFrame([row]).iloc[0]

        opt = optimize_family(r, artifacts, max_cut=max_cut, n_steps=n_steps)

        original_total = (
            housing + utilities + groceries + transport + insurance + childcare +
            healthcare + entertainment + dining_out + clothing + education +
            debt_min + savings + investment + unexpected + other_variable
        )
        cashflow_original = income - original_total

        st.subheader("2. AI Results")

        if not opt["feasible"]:
            st.error("Cannot reach positive cashflow with allowed cuts.")
            log_infeasible_case(
                mode_label, income, cashflow_original,
                opt["pred_cashflow_after"], opt["cut_fraction"]
            )
            st.write(f"Best attempt (MLP-estimated cashflow): ${opt['pred_cashflow_after']:.0f}")
            st.write(f"Max cut used: {opt['cut_fraction']*100:.1f}%")
            return

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Before")
            st.metric("Cashflow", f"${cashflow_original:,.0f}")

        with colB:
            st.markdown("### After (Positive Cashflow)")
            st.metric("Cashflow", f"${opt['pred_cashflow_after']:,.0f}")
            st.write(f"Cut used: {opt['cut_fraction']*100:.1f}%")

        st.subheader("3. Category Changes")
        orig = opt["original_features"]
        best = opt["optimized_features"]

        rows = []
        labels = {
            "utilities": "Utilities",
            "groceries": "Groceries",
            "transport": "Transportation",
            "dining_out": "Dining Out",
            "entertainment": "Entertainment",
            "clothing": "Clothing",
            "other_variable": "Misc / Impulse",
            "unexpected_spend": "Unexpected Cushion",
        }

        for c in FLEX_CATEGORIES:
            rows.append({
                "Category": labels[c],
                "Before ($)": orig[c],
                "After ($)": best[c],
                "Δ ($)": best[c] - orig[c],
                "Δ (%)": (best[c] / orig[c] - 1) * 100,
            })

        st.dataframe(pd.DataFrame(rows).style.format({
            "Before ($)": "{:,.0f}",
            "After ($)": "{:,.0f}",
            "Δ ($)": "{:,.0f}",
            "Δ (%)": "{:,.1f}%",
        }))

        st.subheader("4. Flex Budgets for 3 / 6 / 12 Months")

        horizon = []
        for c in FLEX_CATEGORIES:
            m = best[c]
            horizon.append({
                "Category": labels[c],
                "Monthly": m,
                "3M": m * 3,
                "6M": m * 6,
                "12M": m * 12,
            })

        horizon_df = pd.DataFrame(horizon)
        st.dataframe(
            horizon_df.style.format({
                "Monthly": "{:,.0f}",
                "3M": "{:,.0f}",
                "6M": "{:,.0f}",
                "12M": "{:,.0f}",
            })
        )


# ============================================================
# PART 2 – 24-Month Trend + LSTM Risk Dashboard
# ============================================================

@st.cache_data(show_spinner=False)
def load_trend_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "family_id" not in df.columns:
        df["family_id"] = 1
    if "month_index" not in df.columns:
        df["month_index"] = np.arange(1, len(df) + 1)
    df = df.sort_values(["family_id", "month_index"])
    return df


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def _load_lstm_and_scaler_internal(
    model_path: str = MODEL_FILE,
    scaler_path: str = SCALER_FILE,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    ckpt = torch.load(model_path, map_location=device)

    feature_cols = ckpt["feature_cols"]
    input_size = ckpt["input_size"]
    hidden_size = ckpt["hidden_size"]
    num_layers = ckpt["num_layers"]
    sequence_length = ckpt["sequence_length"]

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler, feature_cols, sequence_length, device


@st.cache_resource(show_spinner=True)
def get_lstm_model_and_scaler():
    """
    Cached LSTM + scaler load.
    Runs once per server start; reused for all requests.
    """
    return _load_lstm_and_scaler_internal()


@torch.no_grad()
def predict_risk_for_family_sequence(
    family_df: pd.DataFrame,
    current_month: int,
    model: nn.Module,
    scaler,
    feature_cols,
    sequence_length: int,
    device: str = "cpu",
    threshold: float = 0.5,
):
    fam_hist = (
        family_df[family_df["month_index"] <= current_month]
        .sort_values("month_index")
        .copy()
    )

    if len(fam_hist) < sequence_length:
        return {"risk_prob": None, "pred_label": None, "enough_history": False}

    seq_df = fam_hist.tail(sequence_length)

    missing_features = [c for c in feature_cols if c not in seq_df.columns]
    if missing_features:
        return {"risk_prob": None, "pred_label": None, "enough_history": False}

    X_seq = seq_df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_seq)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    prob_risk = float(probs[0, 1].item())
    pred_label = int(prob_risk >= threshold)

    return {"risk_prob": prob_risk, "pred_label": pred_label, "enough_history": True}


def trend_dashboard_page(mode: str):
    st.header("📈 24-Month Spending Trend & Risk Dashboard")

    if mode == "Demo Mode":
        st.markdown(
            """
            **Demo Mode – Multi-Family Trend Explorer**

            - Uses synthetic multi-family dataset by default  
            - Great for demos, workshops, investors  
            - Optionally override with your own CSV
            """
        )
    else:
        st.markdown(
            """
            **User Mode – Your Own Trend History**

            - Upload your own 12–24 month history  
            - AI estimates risk of **negative cashflow in the next 3 months**  
            - Shows trends and category breakdowns for your data  
            """
        )

    st.sidebar.subheader("Trend Data & Filters")

    uploaded = st.sidebar.file_uploader(
        "Upload spending trend CSV",
        type=["csv"],
        help="Recommended: month_index, income, all spending categories, cashflow.",
        key="trend_upload",
    )

    # Mode-dependent behavior for data source
    if mode == "Demo Mode":
        if uploaded is not None:
            df = load_trend_data(uploaded)
            st.sidebar.success("Using uploaded file (Demo override).")
        else:
            if os.path.exists(DATA_FILE_DEFAULT):
                df = load_trend_data(DATA_FILE_DEFAULT)
                st.sidebar.info(f"Using built-in demo file: {DATA_FILE_DEFAULT}")
            else:
                st.error(
                    f"Demo file {DATA_FILE_DEFAULT} not found. "
                    f"Please upload a CSV to continue."
                )
                return
    else:  # User Mode
        if uploaded is None:
            st.warning(
                "In **User Mode**, you must upload your own CSV to see the trend dashboard."
            )
            return
        else:
            df = load_trend_data(uploaded)
            st.sidebar.success("Using your uploaded file.")

    model_available = False
    model = scaler = feature_cols = None
    seq_len = 12
    device = "cpu"

    try:
        model, scaler, feature_cols, seq_len, device = get_lstm_model_and_scaler()
        model_available = True
    except Exception as e:
        st.sidebar.warning(
            f"LSTM model not available ({e}). "
            f"Dashboard will fall back to label-based or simple views."
        )

    family_ids = sorted(df["family_id"].unique())
    family_id = st.sidebar.selectbox("Select family", family_ids, key="trend_family")

    family_df = df[df["family_id"] == family_id].sort_values("month_index").copy()

    min_month = int(family_df["month_index"].min())
    max_month = int(family_df["month_index"].max())
    current_month = st.sidebar.slider(
        "Current month (relative index)",
        min_value=min_month,
        max_value=max_month,
        value=max_month,
        step=1,
        key="trend_month",
    )

    st.sidebar.markdown("---")
    st.sidebar.write("**Legend**")
    st.sidebar.write("- 🔴 Out-of-norm or risky behavior")
    st.sidebar.write("- 🟢 Within normal or safe range")

    current_row = family_df[family_df["month_index"] == current_month]
    if current_row.empty:
        st.error("No data for selected month.")
        return
    current_row = current_row.iloc[0]

    hist_df = family_df[family_df["month_index"] <= current_month].copy()

    st.subheader(f"1. Overview for Family #{family_id}, Month {current_month}")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Current Income ($)", f"{current_row['income']:,.0f}")
        avg_income = hist_df["income"].mean()
        st.caption(f"Avg income (to date): ${avg_income:,.0f}")

    with col_b:
        st.metric("Current Cashflow ($)", f"{current_row['cashflow']:,.0f}")
        avg_cf = hist_df["cashflow"].mean()
        st.caption(f"Avg cashflow (to date): ${avg_cf:,.0f}")

    with col_c:
        out_of_norm_any = int(current_row.get("out_of_norm_any", 0))
        out_cat = str(current_row.get("out_of_norm_category", "")) \
            if "out_of_norm_category" in current_row.index else ""

        st.write("Spending vs Norm")
        if out_of_norm_any == 1 and out_cat:
            status_badge("OUT OF NORM", "#c0392b")
            st.caption(f"Main anomaly category (from dataset): **{out_cat}**")
        elif "out_of_norm_any" in family_df.columns:
            status_badge("WITHIN NORM", "#27ae60")
            st.caption("Dataset indicates no major anomaly this month.")
        else:
            status_badge("NO ANOMALY FLAGS PROVIDED", "#7f8c8d")
            st.caption(
                "Your CSV does not include out-of-norm flags. "
                "The model can still analyze risk if required features are present."
            )

    with col_d:
        st.write("Cashflow Risk (Next 3 Months)")

        if model_available:
            risk_info = predict_risk_for_family_sequence(
                family_df=family_df,
                current_month=current_month,
                model=model,
                scaler=scaler,
                feature_cols=feature_cols,
                sequence_length=seq_len,
                device=device,
                threshold=0.5,
            )

            if not risk_info["enough_history"]:
                status_badge("MODEL NOT APPLICABLE", "#7f8c8d")
                st.caption(
                    f"Either not enough months (need ≥{seq_len}) or some required "
                    f"model features are missing in your dataset."
                )
            else:
                risk_prob = risk_info["risk_prob"]
                pred_label = risk_info["pred_label"]

                if pred_label == 1:
                    status_badge("HIGH RISK", "#e67e22")
                else:
                    status_badge("LOW / NORMAL", "#2980b9")

                st.caption(
                    f"Model-estimated probability of negative cashflow in the next 3 months: "
                    f"**{risk_prob*100:.1f}%**"
                )

            if "next_3_months_cashflow_risk" in current_row.index:
                true_label = int(current_row.get("next_3_months_cashflow_risk", 0))
                st.caption(f"Label in dataset (for comparison): `{true_label}`")
        else:
            if "next_3_months_cashflow_risk" in current_row.index:
                risk_label = int(current_row.get("next_3_months_cashflow_risk", 0))
                if risk_label == 1:
                    status_badge("HIGH RISK (label)", "#e67e22")
                else:
                    status_badge("LOW / NORMAL (label)", "#2980b9")
                st.caption(
                    "Using label-based risk because the LSTM model files are not available."
                )
            else:
                status_badge("NO RISK LABELS / MODEL", "#7f8c8d")
                st.caption(
                    "No model and no risk label column found. "
                    "Upload a model or a labeled dataset for full functionality."
                )

        neg_trend = int(current_row.get("negative_trend_warning", 0)) \
            if "negative_trend_warning" in current_row.index else 0
        if neg_trend == 1:
            st.caption("⚠️ Dataset indicates cashflow trend is deteriorating.")
        else:
            st.caption("No explicit trend-warning flag in this row (or it's 0).")

    st.subheader("2. Income vs Total Spending (History to Current Month)")

    line_df = hist_df[["month_index", "income", "total_spend"]].melt(
        id_vars="month_index", var_name="type", value_name="value"
    )

    line_chart = (
        alt.Chart(line_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_index:O", title="Month"),
            y=alt.Y("value:Q", title="Amount ($)"),
            color=alt.Color("type:N", title=""),
            tooltip=["month_index", "type", "value"],
        )
        .properties(height=300)
    )
    st.altair_chart(line_chart, use_container_width=True)

    st.subheader("3. Cashflow Trend and Negative Months")

    cf_chart = (
        alt.Chart(hist_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_index:O", title="Month"),
            y=alt.Y("cashflow:Q", title="Cashflow ($)"),
            tooltip=["month_index", "cashflow"],
        )
        .properties(height=300)
    )

    neg_points = (
        alt.Chart(hist_df[hist_df["cashflow"] < 0])
        .mark_point(size=80, filled=True, color="red")
        .encode(
            x="month_index:O",
            y="cashflow:Q",
            tooltip=["month_index", "cashflow"],
        )
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(
        y="y:Q"
    )

    st.altair_chart(cf_chart + neg_points + zero_line, use_container_width=True)

    st.subheader(f"4. Current Month Category Profile (Month {current_month})")

    cat_values = {}
    for cat in CATEGORIES_TREND:
        if cat in current_row.index:
            cat_values[cat] = float(current_row[cat])

    cat_df = pd.DataFrame(
        {"category": list(cat_values.keys()), "amount": list(cat_values.values())}
    ).sort_values("amount", ascending=False)

    bar_chart = (
        alt.Chart(cat_df)
        .mark_bar()
        .encode(
            x=alt.X("amount:Q", title="Spending ($)"),
            y=alt.Y("category:N", sort="-x", title="Category"),
            tooltip=["category", "amount"],
        )
        .properties(height=400)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    if "out_of_norm_any" in family_df.columns and out_of_norm_any == 1 and out_cat:
        st.markdown(
            f"- 🔴 **Out-of-norm alert (from dataset):** `{out_cat}` is significantly above this family's usual pattern.\n"
            f"- Suggest: coaching text like _'Consider reviewing recent {out_cat} expenses.'_"
        )
    elif "out_of_norm_any" in family_df.columns:
        st.markdown("- 🟢 Dataset indicates no specific category is anomalous this month.")
    else:
        st.markdown(
            "- ℹ️ Your dataset does not include anomaly flags; category profile above still shows where money is going."
        )

    st.subheader("5. Last 6 Months – Quick Audit Table")

    recent_df = family_df[family_df["month_index"] <= current_month].tail(6).copy()
    display_cols = [
        "month_index", "income", "total_spend", "cashflow",
        "out_of_norm_any", "out_of_norm_category",
        "negative_cashflow_flag", "negative_trend_warning",
        "next_3_months_cashflow_risk",
    ]
    cols_to_use = [c for c in display_cols if c in recent_df.columns]

    st.dataframe(
        recent_df[cols_to_use]
        .style.format({"income": "{:,.0f}", "total_spend": "{:,.0f}", "cashflow": "{:,.0f}"})
    )


# ============================================================
# MAIN – App navigation + Mode toggle
# ============================================================

def main():
    st.set_page_config(
        page_title="Smart Spending Guard – Combined App",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🛡️ Smart Spending Guard")

    mode = st.sidebar.radio(
        "Mode:",
        ["Demo Mode", "User Mode"],
        help=(
            "Demo Mode: uses built-in synthetic data (good for demos, workshops).\n"
            "User Mode: expects your real CSV for trend; planner uses your own inputs."
        ),
        key="global_mode",
    )

    page = st.sidebar.radio(
        "Select view:",
        ["Monthly Smart Planner", "24-Month Trend & Risk Dashboard"],
        key="global_page",
    )

    if page == "Monthly Smart Planner":
        smart_planner_page(mode)
    else:
        trend_dashboard_page(mode)


if __name__ == "__main__":
    main()
