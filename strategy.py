#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Uses Camarilla pivot levels (calculated from previous 1d OHLC) for mean reversion entries.
# Long when price touches S3 level and closes above it with 1d uptrend, short when touches R3 with 1d downtrend.
# Works in ranging markets (mean reversion) and trending markets (pullbacks to pivot levels).
# Target: 50-150 total trades over 4 years (12-38/year). Timeframe: 4h, HTF: 1d.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S3 = Pivot - 1.1 * Range / 2
    # S2 = Pivot - 0.75 * Range / 2
    # S1 = Pivot - 0.5 * Range / 2
    # R1 = Pivot + 0.5 * Range / 2
    # R2 = Pivot + 0.75 * Range / 2
    # R3 = Pivot + 1.1 * Range / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    s3_1d = pivot_1d - 1.1 * range_1d / 2
    r3_1d = pivot_1d + 1.1 * range_1d / 2
    
    # Calculate 20-period EMA on 1d for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            continue
        
        # Long entry: price touches S3 and closes above it + 1d uptrend + volume confirmation
        if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and
            close[i] > ema_20_1d_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches R3 and closes below it + 1d downtrend + volume confirmation
        elif (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and
              close[i] < ema_20_1d_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price reverses to opposite pivot level or crosses 1d EMA
        elif position == 1 and (high[i] >= r3_aligned[i] or close[i] < ema_20_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= s3_aligned[i] or close[i] > ema_20_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_1dEMA"
timeframe = "4h"
leverage = 1.0