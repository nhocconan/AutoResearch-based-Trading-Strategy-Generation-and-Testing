# 1D_WeeklyPivot_Crossover_Trend_Filter
# Uses weekly pivot levels (PP, R1, S1) from 1w timeframe for entry/exit signals
# Trend filter: price vs 50-period EMA on 1d timeframe
# Volume confirmation: current volume > 1.5x 20-period average
# Designed to work in both bull and bear markets by following weekly pivot structure with trend filter
# Target: 20-50 trades per year to minimize fee drag

#!/usr/bin/env python3
name = "1D_WeeklyPivot_Crossover_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    
    # Align weekly pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day EMA50 to daily timeframe (no alignment needed as same timeframe)
    ema50_1d_aligned = ema50_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > avg_volume * 1.5
        else:
            volume_confirm = False
        
        if position == 0:
            # Enter long: price crosses above R1 + uptrend + volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below S1 + downtrend + volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below PP
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above PP
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals