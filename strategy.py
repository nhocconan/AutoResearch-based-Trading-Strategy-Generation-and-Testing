#!/usr/bin/env python3
"""
1d_WeeklyCamarilla_R1S1_Breakout_WeeklyTrend
Hypothesis: Weekly Camarilla R1/S1 breakout on daily timeframe with weekly EMA50 trend filter and volume confirmation (>1.5x 20-period MA). 
Long when price breaks above weekly R1 in weekly uptrend with volume spike. Short when price breaks below weekly S1 in weekly downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. 
Designed to work in both bull and bear markets by following the weekly trend.
Target: 20-50 trades over 4 years (5-12/year).
"""

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
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior weekly OHLC
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift by 1 to use prior week's OHLC for current week's levels
    close_1w_prev = np.roll(close_1w, 1)
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev[0] = np.nan
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    
    # Camarilla R1, S1, R3, S3 levels
    camarilla_range = high_1w_prev - low_1w_prev
    r1 = close_1w_prev + camarilla_range * 1.1 / 12
    s1 = close_1w_prev - camarilla_range * 1.1 / 12
    r3 = close_1w_prev + camarilla_range * 1.1 / 4
    s3 = close_1w_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA + 20 for volume MA + 1 for weekly shift)
    start_idx = 71
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with weekly uptrend and volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with weekly downtrend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R3 (strong reversal) OR weekly trend changes to downtrend
            if (close[i] < r3_aligned[i] or not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S3 (strong reversal) OR weekly trend changes to uptrend
            if (close[i] > s3_aligned[i] or not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyCamarilla_R1S1_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0