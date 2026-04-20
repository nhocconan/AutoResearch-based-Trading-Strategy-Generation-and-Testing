#!/usr/bin/env python3
# 1h_4h_1D_Pivot_R1_S1_Breakout_V1
# Hypothesis: Trade breakouts at 1d Camarilla R1/S1 levels with volume confirmation, using 4h trend filter to avoid false breakouts.
# In 4h uptrend (price > EMA20), long breakouts above R1; in 4h downtrend (price < EMA20), short breakdowns below S1.
# Uses volume spike to confirm breakout strength. Targets 15-30 trades/year by requiring confluence of level, volume, and trend.
# Works in both bull and bear markets due to adaptive trend filter.

name = "1h_4h_1D_Pivot_R1_S1_Breakout_V1"
timeframe = "1h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1, R1
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    
    # Align 1d levels to 1h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # 4h uptrend: long breakout above R1
            if close[i] > ema_20_aligned[i]:
                if (close[i] > r1_aligned[i] * 1.002 and 
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.20
                    position = 1
            # 4h downtrend: short breakdown below S1
            elif close[i] < ema_20_aligned[i]:
                if (close[i] < s1_aligned[i] * 0.998 and 
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: price returns below EMA20 or breaks below S1
            if close[i] < ema_20_aligned[i] or close[i] < s1_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns above EMA20 or breaks above R1
            if close[i] > ema_20_aligned[i] or close[i] > r1_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals