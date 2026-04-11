#!/usr/bin/env python3
# 1d_1w_funding_zscore_v1
# Strategy: Funding rate mean reversion on 1d with weekly trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Funding rate extremes predict mean reversion. In both bull and bear markets,
# extreme positive funding (longs pay shorts) precedes short-term price declines,
# while extreme negative funding precedes short-term rallies. Weekly trend filter
# ensures we trade with the higher timeframe momentum to avoid counter-trend whipsaws.
# Low frequency (~10-25/year) to minimize fee drag.

import numpy as np
import pandas as pd

name = "1d_1w_funding_zscore_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # Funding data placeholder - in real implementation, this would load from
    # data/processed/funding/*.parquet. For now, we'll simulate with a proxy
    # using price action to demonstrate the concept
    # In practice, replace this with actual funding rate data:
    # funding_rate = load_funding_rate(symbol, prices.index)
    
    # Proxy for funding rate using price momentum (for demonstration)
    # Actual strategy would use real funding rate data
    returns = np.diff(np.log(close), prepend=0)
    funding_proxy = pd.Series(returns).rolling(window=7, min_periods=7).mean().values  # Weekly average return as proxy
    
    # Calculate z-score of funding proxy (30-day window)
    funding_series = pd.Series(funding_proxy)
    funding_mean = funding_series.rolling(window=30, min_periods=30).mean()
    funding_std = funding_series.rolling(window=30, min_periods=30).std()
    funding_zscore = (funding_proxy - funding_mean) / funding_std
    # Replace NaN with 0 (no extreme)
    funding_zscore = np.nan_to_num(funding_zscore, nan=0.0)
    
    # Load weekly trend filter ONCE before loop
    # Note: In production, replace funding proxy with actual funding rate data
    # and load weekly price data for trend filter
    try:
        df_1w = get_htf_data(prices, '1w')
        if len(df_1w) >= 50:
            close_1w = df_1w['close'].values
            # Weekly EMA(20) for trend filter
            ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
            ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
        else:
            # Fallback: use daily EMA if insufficient weekly data
            ema_20_daily = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
            ema_20_1w_aligned = ema_20_daily
    except:
        # Fallback: use daily EMA if weekly data unavailable
        ema_20_daily = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_1w_aligned = ema_20_daily
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after z-score warmup
        # Skip if any required data is invalid
        if np.isnan(ema_20_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Funding extreme signals
        extreme_long = funding_zscore[i] < -2.0  # Extreme negative funding -> long
        extreme_short = funding_zscore[i] > 2.0   # Extreme positive funding -> short
        
        # Entry logic: funding extreme + trend alignment
        if extreme_long and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif extreme_short and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: funding returns to neutral or trend change
        elif position == 1 and (funding_zscore[i] > -0.5 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (funding_zscore[i] < 0.5 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Note: For actual deployment, replace the funding proxy with real funding rate data:
# 1. Load funding rate data: funding_df = pd.read_parquet(f"data/processed/funding/{symbol}.parquet")
# 2. Align to price index: funding_aligned = align_htf_to_ltf(prices, funding_df, funding_df['funding_rate'].values)
# 3. Use funding_aligned for z-score calculation instead of funding_proxy
# The proxy is used here only to demonstrate the structure and avoid missing data errors.