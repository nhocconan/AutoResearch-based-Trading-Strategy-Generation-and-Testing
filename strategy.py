#!/usr/bin/env python3
# 12H_1D_1W_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Use daily Camarilla R1/S1 levels with 1d EMA50 trend filter and weekly volume confirmation
# for 12h timeframe trading. Weekly volume filter reduces false breakouts, while daily trend filter
# ensures alignment with higher timeframe momentum. Designed for 12-37 trades/year with low turnover.

name = "12H_1D_1W_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot and Camarilla levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = pivot + range_ * 1.1 / 4  # R1 = pivot + (range * 1.1 / 4)
    s1 = pivot - range_ * 1.1 / 4  # S1 = pivot - (range * 1.1 / 4)
    
    # Get weekly data for volume confirmation (weekly average volume)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    # Calculate 4-week average volume for weekly timeframe
    volume_avg_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    
    # Get daily data for EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_avg_1w)
    
    # Current 12h volume compared to weekly average (volume confirmation)
    # Use current volume > 1.5x weekly average volume for confirmation
    volume_confirm = volume > (volume_avg_1w_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_avg_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above daily EMA50 + volume confirmation
            if (close[i] > r1_aligned[i] and close[i] > ema50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + below daily EMA50 + volume confirmation
            elif (close[i] < s1_aligned[i] and close[i] < ema50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily EMA50 (trend change)
            if close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily EMA50 (trend change)
            if close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals