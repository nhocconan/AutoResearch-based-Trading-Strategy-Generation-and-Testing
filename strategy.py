#!/usr/bin/env python3
# 1d_1w_supertrend_reversal_v1
# Strategy: Daily Supertrend reversal with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Supertrend reversals signal momentum shifts. Weekly trend filter (Supertrend) ensures alignment with higher timeframe momentum. Volume confirmation filters weak signals. Works in bull by catching pullbacks in uptrend, and in bear by catching bounces in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_supertrend_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Supertrend for trend filter (10, 3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr = np.zeros_like(tr)
    atr[atr_period] = np.mean(tr[:atr_period+1])
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lowerband[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upperband[i]
        elif direction[i] == 1 and lowerband[i] > supertrend[i-1]:
            supertrend[i] = lowerband[i]
        elif direction[i] == -1 and upperband[i] < supertrend[i-1]:
            supertrend[i] = upperband[i]
        else:
            supertrend[i] = supertrend[i-1]
    
    # Trend filter: 1 for uptrend, 0 for downtrend
    supertrend_trend = (direction == 1).astype(float)
    supertrend_trend_aligned = align_htf_to_ltf(prices, df_1w, supertrend_trend)
    
    # Daily Supertrend for signal (10, 3.0)
    tr1_d = high[1:] - low[1:]
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    
    atr_d = np.zeros_like(tr_d)
    atr_d[atr_period] = np.mean(tr_d[:atr_period+1])
    for i in range(atr_period+1, len(tr_d)):
        atr_d[i] = (atr_d[i-1] * (atr_period-1) + tr_d[i]) / atr_period
    
    hl2_d = (high + low) / 2
    upperband_d = hl2_d + multiplier * atr_d
    lowerband_d = hl2_d - multiplier * atr_d
    
    supertrend_d = np.zeros_like(close)
    direction_d = np.ones_like(close)
    
    supertrend_d[0] = upperband_d[0]
    direction_d[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend_d[i-1]:
            direction_d[i] = 1
        else:
            direction_d[i] = -1
        
        if direction_d[i] == 1 and direction_d[i-1] == -1:
            supertrend_d[i] = lowerband_d[i]
        elif direction_d[i] == -1 and direction_d[i-1] == 1:
            supertrend_d[i] = upperband_d[i]
        elif direction_d[i] == 1 and lowerband_d[i] > supertrend_d[i-1]:
            supertrend_d[i] = lowerband_d[i]
        elif direction_d[i] == -1 and upperband_d[i] < supertrend_d[i-1]:
            supertrend_d[i] = upperband_d[i]
        else:
            supertrend_d[i] = supertrend_d[i-1]
    
    # Signal: Supertrend reversal
    signal_long = (direction_d == 1) & (direction_d[:-1] == -1)  # Bullish reversal
    signal_short = (direction_d == -1) & (direction_d[:-1] == 1)  # Bearish reversal
    
    # Prepend False to match length
    signal_long = np.concatenate([[False], signal_long])
    signal_short = np.concatenate([[False], signal_short])
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(supertrend_trend_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trading logic: only trade in direction of weekly trend
        if signal_long[i] and supertrend_trend_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif signal_short[i] and not supertrend_trend_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and not signal_long[i] and supertrend_trend_aligned[i] == 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not signal_short[i] and supertrend_trend_aligned[i] == 1:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals