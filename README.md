# DayTrader - GPU-Accelerated Trading Research

A Jupyter notebook series for day trading research, leveraging an NVIDIA RTX 4090 (24GB VRAM) for ML inference, NLP sentiment analysis, and time series forecasting.

**This is an educational/research project -- not a live trading system.**

## Notebook Series

The notebooks form a progressive learning path, each building on concepts from the prior:

| # | Notebook | Focus | GPU Use |
|---|----------|-------|---------|
| 1 | `DayTrading_DeepDive.ipynb` | Market data, candlesticks, technical indicators, intro LSTM | PyTorch LSTM training |
| 2 | `02_Sentiment_Pipeline.ipynb` | Multi-model NLP sentiment (FinBERT + Twitter-RoBERTa + DistilRoBERTa), entity extraction, per-stock aggregation | HuggingFace inference, NER |
| 3 | `03_Technical_Strategy.ipynb` | Custom BacktestEngine, 5 strategies (EMA Crossover, RSI Reversion, BB Squeeze, VWAP Bounce, ORB), walk-forward analysis | Minimal |
| 4 | `04_Chronos_Forecasting.ipynb` | Amazon Chronos zero-shot time series forecasting, multi-horizon, stock scanner | Chronos model inference |
| 5 | `05_LLM_Analysis.ipynb` | Local LLM financial analysis (Phi-3, Mistral-7B 4-bit), earnings calls, SEC filings, structured output | LLM inference (fp16 + 4-bit) |
| 6 | `06_Backtesting_Engine.ipynb` | Multi-signal composite scoring, regime detection, portfolio backtester, Monte Carlo simulation | Minimal |
| 7 | `07_Paper_Trading.ipynb` | Paper trading system with simulation replayer, signal hub, regime-aware OMS, Alpaca API integration, SQLite persistence | Optional (Chronos/FinBERT) |

## Requirements

- Python 3.10
- NVIDIA GPU with CUDA support (developed on RTX 4090, 24GB VRAM)
- CUDA 12.4+ compatible driver

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install jupyter ipykernel pandas numpy matplotlib seaborn plotly yfinance scikit-learn mplfinance ta
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate sentencepiece chronos-forecasting
pip install feedparser wordcloud quantstats bitsandbytes

# Register Jupyter kernel
python -m ipykernel install --user --name daytrader --display-name "DayTrader (Python 3.10)"

# Launch
jupyter notebook
```

> **Note:** Use the `ta` library for technical indicators, not `pandas-ta` (incompatible with Python 3.10).

## Key Libraries

| Library | Purpose |
|---------|---------|
| **PyTorch** | LSTM training, GPU acceleration |
| **Transformers** | FinBERT, RoBERTa, Phi-3, Mistral-7B |
| **Chronos** | Zero-shot time series forecasting |
| **yfinance** | Historical and intraday market data |
| **ta** | Technical indicators (EMA, RSI, MACD, Bollinger Bands, ATR) |
| **mplfinance** | Candlestick charting |
| **QuantStats** | Portfolio analytics and tearsheets |

## Hardware

- **Primary GPU:** NVIDIA RTX 4090 (24GB VRAM) -- runs models up to ~13B quantized or ~7B at fp16
- **Secondary GPU:** NVIDIA RTX 2070 SUPER (8GB) -- available but not used by notebooks

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Day trading involves substantial risk of loss. Past performance does not predict future results.
