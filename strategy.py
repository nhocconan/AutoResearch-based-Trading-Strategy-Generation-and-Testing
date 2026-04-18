#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume_Filtered
Hypothesis: Uses daily Camarilla pivot points (R1, S1) with weekly trend filter (price > weekly EMA20 for long, < weekly EMA20 for short) and volume confirmation. Designed for low trade frequency (~10-20/year) with high-probability breakouts in both bull and bear markets. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
"""

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
    
    # Daily Camarilla pivot points (based on previous day)
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    # We use previous day's high, low, close to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_width = 1.1 * (prev_high - prev_low) / 12.0
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Weekly trend filter: EMA20 on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema20_1w = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= 20:
        k = 2 / (20 + 1)
        for i in range(20, len(weekly_close)):
            if i == 20:
                ema20_1w[i] = np.mean(weekly_close[i-20:i+1])
            else:
                ema20_1w[i] = weekly_close[i] * k + ema20_1w[i-1] * (1 - k)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close above R1 with weekly uptrend and volume spike
            if close[i] > r1[i] and close[i] > ema20_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below S1 with weekly downtrend and volume spike
            elif close[i] < s1[i] and close[i] < ema20_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below S1 or weekly trend turns down
            if close[i] < s1[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above R1 or weekly trend turns up
            if close[i] > r1[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume_Filtered"
timeframe = "1d"
leverage = 1.0