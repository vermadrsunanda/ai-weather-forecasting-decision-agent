
🌦️ AI Weather Forecasting Decision Agent

An AI-powered forecasting and decision intelligence application that combines classical time-series forecasting models with a Large Language Model (LLM)–driven decision layer to translate predictions into actionable, stakeholder-ready insights.

This project demonstrates how predictive analytics can be elevated into decision-grade AI systems—a pattern commonly used in enterprise, consulting, and sustainability-focused solutions.
Project Overview

Traditional forecasting systems focus on predicting numbers.
This agent goes a step further by answering:

What do these forecasts mean for planning, operations, and sustainability decisions?

The application:

Uses validated statistical models for forecasting

Uses an LLM only for orchestration and explanation

Ensures numerical reliability while enabling natural-language interaction
✨ Key Features
📈 Forecasting

Daily time-series forecasting (temperature or similar metrics)

Prophet-based forecasting (with automatic fallback if unavailable)

Trend detection and uncertainty bounds

🤖 AI Decision Agent

Natural-language task execution (e.g. “Forecast 14 days and highlight anomalies”)

Tool-calling architecture (LLM orchestrates Python functions)

Decision-oriented outputs:

Trend summary

Business / sustainability implications

Recommended actions

🧩 GenAI Explanations (Optional)

Converts forecast summaries into plain-English insights

Designed for non-technical stakeholders

Uses OpenAI API securely via environment variables

📊 Interactive UI

Streamlit-based dashboard

CSV upload (date + numeric value)

Forecast visualization

Quick prompt buttons for agent tasks
