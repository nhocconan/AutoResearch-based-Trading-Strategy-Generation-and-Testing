#!/usr/bin/env python3
"""
1h_4h_1d_Pivot_R1S1_Breakout_Volume
Hypothesis: On 1h timeframe, trade breakouts from 1d-derived R1/S1 levels with volume spike confirmation.
Use 4h trend filter (EMA50) to avoid counter-trend trades. R1/S1 are tighter than R4/S4, increasing trade frequency
but with volume and trend filters to maintain quality. Targets 60-150 total trades over 4 years.
Works in bull/bear by aligning with 4h trend direction. Uses discrete position sizing (0.20) to minimize churn.
"""

name = "1h_4h_1d_Pivot_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d pivot and R1/S1 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R1 and S1 levels
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    
    # Align 1d levels to 1h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R1, volume spike, and price above 4h EMA50 (uptrend)
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below S1, volume spike, and price below 4h EMA50 (downtrend)
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 or trend reversal (below EMA50)
            if close[i] < s1_aligned[i] * 0.998 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price above R1 or trend reversal (above EMA50)
            if close[i] > r1_aligned[i] * 1.002 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals