# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jupyter notebook-based day trading research environment leveraging an RTX 4090 GPU (24GB VRAM) for ML inference, sentiment analysis, and time series forecasting. Educational/research project -- not a live trading system.

## Environment Setup

```bash
source venv/bin/activate
jupyter notebook  # launches on the "daytrader" kernel (Python 3.10)
```

The `venv/` contains all dependencies. The Jupyter kernel is registered as `daytrader`. If dependencies need reinstalling:

```bash
python3 -m venv venv && source venv/bin/activate
pip install jupyter ipykernel pandas numpy matplotlib seaborn plotly yfinance scikit-learn mplfinance ta
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate sentencepiece chronos-forecasting
pip install feedparser wordcloud quantstats bitsandbytes
python -m ipykernel install --user --name daytrader --display-name "DayTrader (Python 3.10)"
```

Note: `pandas-ta` is NOT compatible with Python 3.10. Use `ta` (technical analysis library) instead.

## Notebook Series Architecture

The notebooks form a progressive learning series, each building on concepts from the prior:

| Notebook | Focus | GPU Use |
|----------|-------|---------|
| `DayTrading_DeepDive.ipynb` | Overview: market data, candlesticks, indicators, intro LSTM | PyTorch LSTM training |
| `02_Sentiment_Pipeline.ipynb` | Multi-model NLP sentiment (FinBERT + Twitter-RoBERTa + DistilRoBERTa), entity extraction, per-stock aggregation | HuggingFace inference, NER |
| `03_Technical_Strategy.ipynb` | Custom `BacktestEngine`, 5 strategies (EMA Crossover, RSI Reversion, BB Squeeze, VWAP Bounce, ORB), walk-forward analysis | Minimal (data processing) |
| `04_Chronos_Forecasting.ipynb` | Amazon Chronos zero-shot time series forecasting, multi-horizon, stock scanner, combined technical+forecast signals | Chronos model inference |
| `05_LLM_Analysis.ipynb` | Local LLM financial analysis (Phi-3, Mistral-7B 4-bit), earnings calls, SEC filings, structured JSON output, morning briefs | LLM inference (fp16 + 4-bit quantized) |
| `06_Backtesting_Engine.ipynb` | Multi-signal composite scoring, regime detection, portfolio backtester, Monte Carlo simulation, walk-forward validation, parameter sensitivity | Minimal (data processing) |

## Key Patterns

- **yfinance data handling**: DataFrames may have MultiIndex columns. Always flatten with `df.columns = df.columns.get_level_values(0)` before use.
- **HuggingFace models**: Use `device=0` for GPU or `device_map="cuda"` for Chronos. Models download on first run and cache in `~/.cache/huggingface/`.
- **Chronos**: Use `ChronosBoltPipeline` (from `chronos` package) for speed, `ChronosPipeline` for accuracy. Bolt models are preferred for real-time use.
- **Technical indicators**: Use the `ta` library (e.g., `from ta.trend import EMAIndicator`), NOT `pandas-ta`.
- **BacktestEngine** (notebook 03): Enters on next bar's open to avoid look-ahead bias. Strategies are functions with signature `strategy_fn(df, i, position) -> signal_dict | None`.
- **Sentiment normalization**: FinBERT/Twitter-RoBERTa/DistilRoBERTa all output different label names but map to positive/negative/neutral. The `normalize_sentiment()` function in notebook 02 standardizes to a -1 to +1 scale.
- **LLM loading** (notebook 05): Use `AutoModelForCausalLM` with `torch_dtype=torch.float16` for smaller models (Phi-3). Use `BitsAndBytesConfig(load_in_4bit=True)` for 7B+ models (Mistral). Always `trust_remote_code=True` for Phi-3.
- **Composite signals** (notebook 06): All signals normalized to -1 (bearish) to +1 (bullish). `RegimeDetector` uses ADX (>25 trending, <20 ranging) for adaptive weighting. `PortfolioBacktester` trades multiple stocks with position limits and risk management.

## Hardware

- Primary GPU: NVIDIA RTX 4090 (24GB VRAM) -- `cuda:0`
- Secondary GPU: NVIDIA RTX 2070 SUPER (8GB) -- `cuda:1` (not used by notebooks)
- The 4090 can run models up to ~13B parameters quantized or ~7B at fp16
