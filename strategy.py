#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h/1d Trend and Volume Confirmation
# Hypothesis: RSI pullbacks in trending markets provide high-probability entries.
# Use 4h EMA50 and 1d EMA200 for trend alignment (works in bull/bear).
# Volume spike confirms momentum. RSI < 40 for long, > 60 for short.
# Session filter (08-20 UTC) reduces noise. Target 15-30 trades/year.

name = "1h_rsi_pullback_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA50 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA200 for long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 60 or trend turns bearish
            if rsi[i] > 60 or close[i] < ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 40 or trend turns bullish
            if rsi[i] < 40 or close[i] > ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI < 40 (pullback) + volume spike + uptrend (above both EMAs)
            if rsi[i] < 40 and vol_spike[i] and close[i] > ema50_4h_aligned[i] and close[i] > ema200_1d_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short: RSI > 60 (pullback) + volume spike + downtrend (below both EMAs)
            elif rsi[i] > 60 and vol_spike[i] and close[i] < ema50_4h_aligned[i] and close[i] < ema200_1d_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals