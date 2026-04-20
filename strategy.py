#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter_v1
Concept: Daily Camarilla pivot breakout with weekly trend filter to reduce false signals.
- Uses daily pivot points (R1, S1) as key support/resistance
- Long when price breaks above R1 with volume confirmation (>1.5x avg) and weekly trend up
- Short when price breaks below S1 with volume confirmation (>1.5x avg) and weekly trend down
- Exit when price returns to central pivot point
- Uses weekly EMA20 for trend filter
- Conservative sizing (0.25) to manage drawdown
- Target: 30-80 trades per symbol over 4 years (7-20/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === Weekly: EMA20 trend filter ===
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Get daily data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        weekly_trend = weekly_ema20_aligned[i]
        close_val = prices['close'].iloc[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(weekly_trend) or np.isnan(pivot_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weekly trend up
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.5
            
            if breakout_long and vol_confirm and close_val > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weekly trend down
            elif close_val < s1_val and vol_confirm and close_val < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below central pivot
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above central pivot
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals