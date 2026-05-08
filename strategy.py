#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    R4 = close_1d + (high_1d - low_1d) * 1.500
    R3 = close_1d + (high_1d - low_1d) * 1.250
    S3 = close_1d - (high_1d - low_1d) * 1.250
    S4 = close_1d - (high_1d - low_1d) * 1.500
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike detection on 6h
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 4 days of 6h bars
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(sma_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in weekly uptrend
            long_breakout = close[i] > r3_aligned[i] and vol_ratio[i] > 2.0
            # Short: price breaks below S3 with volume spike in weekly downtrend
            short_breakout = close[i] < s3_aligned[i] and vol_ratio[i] > 2.0
            
            if long_breakout and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            elif short_breakout and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 or weekly trend turns down
            if close[i] < s3_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 or weekly trend turns up
            if close[i] > r3_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as strong support/resistance. 
# Breakouts with volume spikes indicate institutional interest. 
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Volume spike requirement (>2x average) filters false breakouts.
# Target: 50-150 total trades over 4 years to minimize fee decay.