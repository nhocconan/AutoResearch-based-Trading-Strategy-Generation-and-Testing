#!/usr/bin/env python3
# 1d_Range_Breakout_With_Volume_and_Trend_Filter
# Hypothesis: On daily timeframe, breakouts of the previous day's high/low with volume confirmation
# and trend filter (EMA50) capture institutional moves. Works in bull/bear by following EMA50 trend.
# Target: 15-25 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "1d"
name = "1d_Range_Breakout_With_Volume_and_Trend_Filter"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for range and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Previous day's high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to daily timeframe (previous day's values available at open)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure EMA50 and volume MA ready
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above previous day's high with volume, and above EMA50
            if (high[i] > high_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below previous day's low with volume, and below EMA50
            elif (low[i] < low_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: break below previous day's low (mean reversion)
            if low[i] < low_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: break above previous day's high (mean reversion)
            if high[i] > high_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals