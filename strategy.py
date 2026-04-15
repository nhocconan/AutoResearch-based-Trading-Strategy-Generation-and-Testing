#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion strategy using 4h RSI extremes and 1d trend filter.
# - 4h RSI(14) < 30 for long, > 70 for short (mean reversion in 4h)
# - 1d close above/below 50 EMA for trend filter (avoid counter-trend trades)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Fixed position size: 0.20 (20% of capital) to limit drawdown
# - Target: 15-30 trades/year by using strict 4h RSI thresholds and 1d EMA filter
# - Works in bull/bear: 1d EMA filter ensures we only trade with higher timeframe trend,
#   while 4h RSI captures short-term mean reversion within that trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: RSI(14) for mean reversion signals ===
    close_4h = pd.Series(df_4h['close'].values)
    delta = close_4h.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h = rsi_14_4h.fillna(50).values  # neutral RSI when insufficient data
    
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # === 1d Indicators: EMA(50) for trend filter ===
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 4h RSI < 30 (oversold mean reversion)
        # 2. 1d close above 50 EMA (bullish higher timeframe trend)
        if (rsi_14_4h_aligned[i] < 30 and
            close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 4h RSI > 70 (overbought mean reversion)
        # 2. 1d close below 50 EMA (bearish higher timeframe trend)
        elif (rsi_14_4h_aligned[i] > 70 and
              close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_RSI14_4h_EMA50_1d_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0