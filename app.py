"""
Project 1 Starter Code: AI Weather Forecasting & Decision Support Agent
- Domain: UAE-style daily temperature forecasting (can be adapted to energy demand, CO2, etc.)
- Features:
  1) Upload CSV (date, value)
  2) Forecast with Prophet (fallback to a simple moving average if Prophet not installed)
  3) Plot forecast
  4) "Explain Forecast" using OpenAI / Azure OpenAI (optional; works even if you skip API)
Run:
  pip install streamlit pandas numpy matplotlib
  pip install prophet   # optional but recommended (may be 'prophet' or 'fbprophet' depending on env)
  pip install openai    # optional if you want LLM explanation

Then:
  streamlit run app.py
"""

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# Optional: Prophet forecasting
# -----------------------------
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# -----------------------------
# Optional: OpenAI explanation
# -----------------------------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# -----------------------------
# Helpers
# -----------------------------
def make_sample_csv() -> bytes:
    """Creates a simple sample dataset: daily temperatures for ~180 days."""
    rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180, freq="D")
    # Create a seasonal-ish pattern + noise
    base = 28 + 4 * np.sin(np.linspace(0, 3*np.pi, len(rng)))
    noise = np.random.normal(0, 0.6, len(rng))
    temp = base + noise
    df = pd.DataFrame({"date": rng.strftime("%Y-%m-%d"), "temperature": np.round(temp, 2)})
    return df.to_csv(index=False).encode("utf-8")


def load_and_prepare(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Standardize to Prophet format: ds, y"""
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {date_col} and {value_col}")

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna().sort_values(date_col)

    out = out.rename(columns={date_col: "ds", value_col: "y"})
    # Prophet works best with regular frequency; we’ll assume daily and fill gaps if needed
    out = out.set_index("ds").asfreq("D")
    out["y"] = out["y"].interpolate(limit_direction="both")
    out = out.reset_index()
    return out


def forecast_with_prophet(train_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Forecast using Prophet; returns dataframe with ds, yhat, yhat_lower, yhat_upper."""
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.9,
    )
    m.fit(train_df)

    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fcst


def forecast_fallback_ma(train_df: pd.DataFrame, horizon_days: int, window: int = 7) -> pd.DataFrame:
    """Fallback forecast: rolling mean with a simple trend extension."""
    df = train_df.copy()
    df["ma"] = df["y"].rolling(window=window, min_periods=1).mean()

    last_date = df["ds"].max()
    last_ma = float(df["ma"].iloc[-1])
    # crude trend: difference between last MA and MA from 14 days ago
    lookback = min(14, len(df)-1)
    trend = (last_ma - float(df["ma"].iloc[-lookback])) / max(lookback, 1)

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    yhat = [last_ma + trend * i for i in range(1, horizon_days + 1)]

    fcst = pd.DataFrame({
        "ds": pd.concat([df["ds"], pd.Series(future_dates)], ignore_index=True),
        "yhat": pd.concat([df["ma"], pd.Series(yhat)], ignore_index=True)
    })
    # naive uncertainty bounds
    resid = (df["y"] - df["ma"]).dropna()
    sigma = float(resid.std()) if len(resid) > 5 else 1.0
    fcst["yhat_lower"] = fcst["yhat"] - 1.64 * sigma
    fcst["yhat_upper"] = fcst["yhat"] + 1.64 * sigma
    return fcst


def plot_forecast(train_df: pd.DataFrame, fcst: pd.DataFrame, title: str = "Forecast"):
    """Matplotlib plot (no fixed colors)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_df["ds"], train_df["y"], label="Actual")
    ax.plot(fcst["ds"], fcst["yhat"], label="Forecast (yhat)")
    ax.fill_between(fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"], alpha=0.2, label="Uncertainty")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig


def summarize_forecast(train_df: pd.DataFrame, fcst: pd.DataFrame, horizon_days: int) -> dict:
    """Create a compact summary for the LLM."""
    last_actual = float(train_df["y"].iloc[-1])
    last_date = train_df["ds"].iloc[-1]

    future = fcst[fcst["ds"] > last_date].head(horizon_days).copy()
    if future.empty:
        future = fcst.tail(horizon_days).copy()

    start_pred = float(future["yhat"].iloc[0])
    end_pred = float(future["yhat"].iloc[-1])
    delta = end_pred - start_pred
    pct = (delta / abs(start_pred) * 100.0) if start_pred != 0 else None

    # detect rough trend
    trend_label = "increasing" if delta > 0.2 else "decreasing" if delta < -0.2 else "stable"

    return {
        "last_observed_date": str(pd.to_datetime(last_date).date()),
        "last_observed_value": round(last_actual, 3),
        "forecast_horizon_days": horizon_days,
        "forecast_start_date": str(pd.to_datetime(future["ds"].iloc[0]).date()),
        "forecast_end_date": str(pd.to_datetime(future["ds"].iloc[-1]).date()),
        "forecast_start_value": round(start_pred, 3),
        "forecast_end_value": round(end_pred, 3),
        "delta_over_horizon": round(delta, 3),
        "percent_change_over_horizon": round(pct, 2) if pct is not None else None,
        "trend": trend_label,
        "min_forecast_value": round(float(future["yhat"].min()), 3),
        "max_forecast_value": round(float(future["yhat"].max()), 3),
    }


def explain_with_openai(summary: dict, domain: str, audience: str, use_azure: bool = False) -> str:
    """
    Uses OpenAI or Azure OpenAI via environment variables.
    - If use_azure=False:
        set OPENAI_API_KEY
        optionally set OPENAI_MODEL (default gpt-4o-mini)
    - If use_azure=True (Azure OpenAI):
        set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
        set AZURE_OPENAI_DEPLOYMENT (model deployment name)
    """
    if not OPENAI_AVAILABLE:
        return "OpenAI SDK not installed. Install with: pip install openai"

    prompt = f"""
You are a data science consultant. Explain this forecast summary in plain English.

Domain: {domain}
Audience: {audience}

Forecast Summary (JSON):
{json.dumps(summary, indent=2)}

Please provide:
1) A 3–5 bullet explanation of the trend and key numbers
2) Practical business impact (2–3 bullets)
3) Recommended actions (3 bullets)
Keep it concise and professional.
""".strip()

    if use_azure:
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

        if not (api_key and endpoint and deployment):
            return ("Azure OpenAI env vars missing. Please set:\n"
                    "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT\n"
                    "Optionally: AZURE_OPENAI_API_VERSION")

        client = OpenAI(
            api_key=api_key,
            base_url=f"{endpoint}/openai/deployments/{deployment}",
            default_query={"api-version": api_version},
        )

        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful forecasting analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            return "OPENAI_API_KEY not set. Add it as an environment variable to enable explanations."

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful forecasting analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Forecasting Decision Agent", layout="wide")

st.title("AI Weather Forecasting & Decision Support Agent")
st.caption("Upload a CSV with a date column + a value column (e.g., temperature). Forecast + optional GenAI explanation.")

with st.sidebar:
    st.header("1) Data")
    st.download_button(
        "Download sample CSV",
        data=make_sample_csv(),
        file_name="sample_weather_temperature.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    st.header("2) Settings")
    horizon_days = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30, step=1)

    domain = st.selectbox("Domain label", ["UAE Weather (Temperature)", "Energy Demand", "CO₂ Emissions", "Other"])
    audience = st.selectbox("Audience", ["Business stakeholder", "Operations manager", "Technical team"])

    st.header("3) GenAI (Optional)")
    use_llm = st.checkbox("Enable AI explanation", value=False)
    use_azure = st.checkbox("Use Azure OpenAI (instead of OpenAI)", value=False)
    st.write("Env vars needed:")
    st.code(
        "OpenAI: OPENAI_API_KEY, (optional) OPENAI_MODEL\n"
        "Azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, (optional) AZURE_OPENAI_API_VERSION"
    )

# Load data
if uploaded is None:
    st.info("Upload a CSV to begin, or download and use the sample CSV from the sidebar.")
    st.stop()

raw = pd.read_csv(uploaded)
st.subheader("Uploaded Data Preview")
st.dataframe(raw.head(15), use_container_width=True)

# Select columns
cols = list(raw.columns)
if len(cols) < 2:
    st.error("Your CSV must have at least 2 columns: a date column and a numeric value column.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    date_col = st.selectbox("Select date column", options=cols, index=0)
with c2:
    value_col = st.selectbox("Select value column (numeric)", options=cols, index=1)

# Prepare
try:
    train_df = load_and_prepare(raw, date_col=date_col, value_col=value_col)
except Exception as e:
    st.error(f"Data preparation error: {e}")
    st.stop()

st.subheader("Prepared Time Series (Daily)")
st.dataframe(train_df.tail(15), use_container_width=True)

# Forecast
st.subheader("Forecast")
if PROPHET_AVAILABLE:
    st.success("Prophet is available ✅ Using Prophet for forecasting.")
    fcst = forecast_with_prophet(train_df, horizon_days=horizon_days)
else:
    st.warning("Prophet not available. Using fallback moving-average forecast. (Install Prophet for better results.)")
    fcst = forecast_fallback_ma(train_df, horizon_days=horizon_days)

fig = plot_forecast(train_df, fcst, title=f"Forecast ({horizon_days} days)")
st.pyplot(fig, clear_figure=True)

# Summary
summary = summarize_forecast(train_df, fcst, horizon_days=horizon_days)
st.subheader("Forecast Summary (for stakeholders / LLM)")
st.json(summary)

# Explain
if use_llm:
    st.subheader("AI Explanation")
    if st.button("Explain forecast in plain English"):
        with st.spinner("Generating explanation..."):
            explanation = explain_with_openai(
                summary=summary,
                domain=domain,
                audience=audience,
                use_azure=use_azure
            )
        st.write(explanation)

st.markdown("---")
st.caption("Next upgrades: add energy-demand optimization, anomaly detection, PDF export, and Azure deployment notes.")
