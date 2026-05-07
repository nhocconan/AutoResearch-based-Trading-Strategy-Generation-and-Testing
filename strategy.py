#!/usr/bin/env python3
"""
6h_Elder_Ray_Cross_1wTrend_v1
Hypothesis: On 6h timeframe, use Elder Ray (Bull/Bear power) crossovers from daily data for entries, filtered by weekly trend (EMA50) and volume spikes. Long when Bull Power crosses above zero with weekly uptrend and volume spike. Short when Bear Power crosses below zero with weekly downtrend and volume spike. Elder Ray captures bull/bear power via EMA13, providing early trend signals. Weekly EMA50 filter ensures alignment with higher timeframe trend. Volume spike confirms conviction. Designed for moderate frequency (target 50-150 total trades over 4 years) to balance opportunity and fee drag on 6h timeframe.
"""
name = "6h_Elder_Ray_Cross_1wTrend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation (EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Daily EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume filter: current volume > 1.5 * 30-period average volume
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 13)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero + weekly uptrend + volume filter
            if (bull_power_aligned[i] > 0 and bull_power_aligned[i-1] <= 0 and 
                close[i] > ema_50_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses below zero + weekly downtrend + volume filter
            elif (bear_power_aligned[i] < 0 and bear_power_aligned[i-1] >= 0 and 
                  close[i] < ema_50_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Elder Ray power crosses back through zero (mean reversion)
            if position == 1 and bull_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals