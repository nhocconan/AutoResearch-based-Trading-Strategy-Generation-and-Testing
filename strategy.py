#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_cci_trend_follow
# Uses weekly CCI (20-period) to determine trend direction on daily chart.
# Long when weekly CCI > 100 and daily close > daily EMA(50).
# Short when weekly CCI < -100 and daily close < daily EMA(50).
# Exit when weekly CCI crosses back to neutral zone (-100 to 100).
# Designed for very low trade frequency (<10 trades/year) to minimize fee drag.
# Works in trending markets via trend following and avoids whipsaws in ranging markets.

name = "1d_1w_cci_trend_follow"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for CCI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly CCI (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    # SMA of typical price
    sma_tp = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI calculation
    cci = (tp_1w - sma_tp) / (0.015 * md)
    
    # Align weekly CCI to daily timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci)
    
    # Daily EMA(50) for trend confirmation
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(cci_aligned[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # Long signal: weekly CCI > 100 and daily close > EMA(50)
        if cci_aligned[i] > 100 and close[i] > ema_50[i] and position != 1:
            position = 1
            signals[i] = 0.30
        # Short signal: weekly CCI < -100 and daily close < EMA(50)
        elif cci_aligned[i] < -100 and close[i] < ema_50[i] and position != -1:
            position = -1
            signals[i] = -0.30
        # Exit conditions: weekly CCI crosses back to neutral zone
        elif position == 1 and cci_aligned[i] < 100:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_aligned[i] > -100:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals