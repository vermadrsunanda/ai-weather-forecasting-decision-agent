"""
AI Weather Forecasting & Decision Support Agent (with LLM Agent mode)

Features:
1) Upload CSV (date, value)
2) Forecast with Prophet (fallback to moving average if Prophet not installed)
3) Plot forecast
4) Explain Forecast (optional)
5) Ask the Weather Agent (LLM orchestrates tools + generates decision-grade response)
   + Quick prompt buttons
Run:
  pip install streamlit pandas numpy matplotlib openai
  pip install prophet   # optional
Then:
  streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
# Optional: OpenAI explanation / agent
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
    base = 28 + 4 * np.sin(np.linspace(0, 3*np.pi, len(rng)))
    noise = np.random.normal(0, 0.6, len(rng))
    temp = base + noise
    df = pd.DataFrame({"date": rng.strftime("%Y-%m-%d"), "temperature": np.round(temp, 2)})
    return df.to_csv(index=False).encode("utf-8")


def load_and_prepare(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Standardize to Prophet format: ds, y (daily)"""
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {date_col} and {value_col}")

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna().sort_values(date_col)

    out = out.rename(columns={date_col: "ds", value_col: "y"})
    out = out.set_index("ds").asfreq("D")
    out["y"] = out["y"].interpolate(limit_direction="both")
    out = out.reset_index()
    return out


def forecast_with_prophet(train_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
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
    """Fallback forecast: rolling mean + crude trend extension."""
    df = train_df.copy()
    df["ma"] = df["y"].rolling(window=window, min_periods=1).mean()

    last_date = df["ds"].max()
    last_ma = float(df["ma"].iloc[-1])
    lookback = min(14, len(df) - 1)
    trend = (last_ma - float(df["ma"].iloc[-lookback])) / max(lookback, 1)

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    yhat = [last_ma + trend * i for i in range(1, horizon_days + 1)]

    fcst = pd.DataFrame({
        "ds": pd.concat([df["ds"], pd.Series(future_dates)], ignore_index=True),
        "yhat": pd.concat([df["ma"], pd.Series(yhat)], ignore_index=True),
    })

    resid = (df["y"] - df["ma"]).dropna()
    sigma = float(resid.std()) if len(resid) > 5 else 1.0
    fcst["yhat_lower"] = fcst["yhat"] - 1.64 * sigma
    fcst["yhat_upper"] = fcst["yhat"] + 1.64 * sigma
    return fcst


def plot_forecast(train_df: pd.DataFrame, fcst: pd.DataFrame, title: str):
    """Safe plotting (convert to numpy arrays to avoid dtype issues)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_df["ds"], train_df["y"], label="Actual")
    ax.plot(fcst["ds"], fcst["yhat"], label="Forecast (yhat)")
    ax.fill_between(
        fcst["ds"],
        fcst["yhat_lower"].to_numpy(),
        fcst["yhat_upper"].to_numpy(),
        alpha=0.2,
        label="Uncertainty"
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig


def summarize_forecast(train_df: pd.DataFrame, fcst: pd.DataFrame, horizon_days: int) -> dict:
    last_actual = float(train_df["y"].iloc[-1])
    last_date = train_df["ds"].iloc[-1]

    future = fcst[fcst["ds"] > last_date].head(horizon_days).copy()
    if future.empty:
        future = fcst.tail(horizon_days).copy()

    start_pred = float(future["yhat"].iloc[0])
    end_pred = float(future["yhat"].iloc[-1])
    delta = end_pred - start_pred
    pct = (delta / abs(start_pred) * 100.0) if start_pred != 0 else None
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


def explain_with_openai(summary: dict, domain: str, audience: str) -> str:
    if not OPENAI_AVAILABLE:
        return "OpenAI SDK not installed. Install with: pip install openai"

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "OPENAI_API_KEY not set. Add it as an environment variable / Streamlit secret."

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
# Agent tools (LLM uses these)
# -----------------------------
def tool_get_forecast_summary(train_df: pd.DataFrame, fcst: pd.DataFrame, horizon_days: int) -> dict:
    summary = summarize_forecast(train_df, fcst, horizon_days=horizon_days)
    return {"summary": summary}


def tool_get_explanation(summary: dict, domain: str, audience: str) -> dict:
    text = explain_with_openai(summary, domain=domain, audience=audience)
    return {"explanation": text}


def run_weather_agent(user_task: str, train_df: pd.DataFrame, fcst: pd.DataFrame, domain: str, audience: str) -> str:
    """
    Proper tool-calling loop:
    1) LLM decides tool call(s)
    2) We execute tool functions in Python
    3) LLM returns final answer using tool outputs
    """
    if not OPENAI_AVAILABLE:
        return "OpenAI SDK not installed. Run: pip install openai"

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "OPENAI_API_KEY is missing. Add it to environment variables or Streamlit secrets."

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Function map for tool execution
    tool_map = {
        "tool_get_forecast_summary": lambda args: tool_get_forecast_summary(
            train_df=train_df,
            fcst=fcst,
            horizon_days=int(args.get("horizon_days", 30))
        ),
        "tool_get_explanation": lambda args: tool_get_explanation(
            summary=args.get("summary", {}),
            domain=args.get("domain", domain),
            audience=args.get("audience", audience),
        ),
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "tool_get_forecast_summary",
                "description": "Get forecast summary for a horizon in days (7-60).",
                "parameters": {
                    "type": "object",
                    "properties": {"horizon_days": {"type": "integer", "minimum": 7, "maximum": 60}},
                    "required": ["horizon_days"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_explanation",
                "description": "Explain forecast summary for stakeholders with business/sustainability implications.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "object"},
                        "domain": {"type": "string"},
                        "audience": {"type": "string"},
                    },
                    "required": ["summary", "domain", "audience"]
                },
            },
        },
    ]

    system = (
        "You are a Weather Forecasting Decision Agent.\n"
        "Rules:\n"
        "- Do NOT invent numbers.\n"
        "- Always use tool outputs for any forecast values.\n"
        "- If horizon is not specified, choose 30 days.\n"
        "- Use tool_get_forecast_summary first, then tool_get_explanation.\n"
        "- Final answer must have 3 sections: Trend Summary, Implications, Recommended Actions.\n"
        "Be concise and professional."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_task},
    ]

    # Step 1: model decides tool calls
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )
    msg = first.choices[0].message

    # Step 2: execute tool calls (if any)
    if getattr(msg, "tool_calls", None):
        messages.append(msg)

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")

            if name in tool_map:
                result = tool_map[name](args)
            else:
                result = {"error": f"Unknown tool: {name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

        # Step 3: final response based on tool outputs
        final = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return final.choices[0].message.content

    # If no tool calls, return whatever it said
    return msg.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Forecasting Decision Agent", layout="wide")
st.title("AI Weather Forecasting & Decision Support Agent")
st.caption("Upload CSV → Forecast → Explain / Ask Agent (LLM orchestrates tools).")

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

    st.header("3) GenAI")
    use_llm_explain = st.checkbox("Enable AI explanation (button below)", value=False)
    st.caption("Set OPENAI_API_KEY to enable LLM features.")

# Must upload a CSV
if uploaded is None:
    st.info("Upload a CSV to begin, or download and use the sample CSV from the sidebar.")
    st.stop()

# Safe CSV read
try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the uploaded file. Please upload a valid CSV. Error: {e}")
    st.stop()

st.subheader("Uploaded Data Preview")
st.dataframe(raw.head(15), use_container_width=True)

cols = list(raw.columns)
if len(cols) < 2:
    st.error("CSV must have at least 2 columns: a date column and a numeric value column.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    date_col = st.selectbox("Select date column", options=cols, index=0)
with c2:
    value_col = st.selectbox("Select value column (numeric)", options=cols, index=1)

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
    st.warning("Prophet not available. Using fallback moving-average forecast.")
    fcst = forecast_fallback_ma(train_df, horizon_days=horizon_days)

fig = plot_forecast(train_df, fcst, title=f"Forecast ({horizon_days} days)")
st.pyplot(fig, clear_figure=True)

# Summary
summary = summarize_forecast(train_df, fcst, horizon_days=horizon_days)
st.subheader("Forecast Summary (for stakeholders / LLM)")
st.json(summary)

# Explain (single-shot)
if use_llm_explain:
    st.subheader("AI Explanation (Single-shot)")
    if st.button("Explain forecast in plain English"):
        with st.spinner("Generating explanation..."):
            explanation = explain_with_openai(summary=summary, domain=domain, audience=audience)
        st.write(explanation)

st.markdown("---")

# Ask the Weather Agent (buttons + text area)
st.subheader("Ask the Weather Agent (LLM Orchestrated)")

if "agent_task" not in st.session_state:
    st.session_state.agent_task = "Forecast next 30 days and explain what it means for operations and sustainability planning."

b1, b2, b3 = st.columns(3)

with b1:
    if st.button("14-day spikes"):
        st.session_state.agent_task = "Forecast next 14 days and highlight unusual spikes or anomalies."

with b2:
    if st.button("Non-technical summary"):
        st.session_state.agent_task = "Summarize the forecast trend for a non-technical manager in simple language."

with b3:
    if st.button("Recommended actions"):
        st.session_state.agent_task = "What actions would you recommend if temperatures rise compared to recent history?"

agent_task = st.text_area("Type your request", key="agent_task", height=90)

if st.button("Run Agent"):
    # Ensure forecast exists (it does here), then run agent
    with st.spinner("Agent is working..."):
        answer = run_weather_agent(
            user_task=agent_task,
            train_df=train_df,
            fcst=fcst,
            domain=domain,
            audience=audience
        )
    st.markdown("### Agent Answer")
    st.write(answer)

st.caption("Tip: Numbers come from the forecasting model, not the LLM. LLM is used for orchestration + explanation.")
