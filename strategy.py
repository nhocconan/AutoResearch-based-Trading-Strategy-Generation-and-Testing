#!/usr/bin/env python3
"""
Hypothesis: On 12h, price respects weekly Pivot (S1/R1) as support/resistance.
We use 1-week Pivot levels (based on prior week's high/low/close) as S1/R1.
We go long when price crosses above R1 with volume > 1.3x average and price above 1d EMA50.
We go short when price crosses below S1 with volume > 1.3x average and price below 1d EMA50.
Exit when price returns to the weekly pivot point (P) or on opposite signal.
Designed for 12h to work in trending and ranging markets with ~15-30 trades per year.
"""

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
    
    # Get 1w data for weekly Pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Pivot levels from prior week's data (avoid look-ahead)
    p_high = df_1w['high'].shift(1).values
    p_low = df_1w['low'].shift(1).values
    p_close = df_1w['close'].shift(1).values
    
    # Weekly Pivot Point (P) and support/resistance levels
    pivot = (p_high + p_low + p_close) / 3
    s1 = 2 * pivot - p_high
    r1 = 2 * pivot - p_low
    s2 = pivot - (p_high - p_low)
    r2 = pivot + (p_high - p_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 (use prior day's close to avoid look-ahead)
    ema_50 = pd.Series(p_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1w, pivot)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1)
    r1_12h = align_htf_to_ltf(prices, df_1w, r1)
    s2_12h = align_htf_to_ltf(prices, df_1w, s2)
    r2_12h = align_htf_to_ltf(prices, df_1w, r2)
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(r1_12h[i]) or 
            np.isnan(s2_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(ema_50_12h[i]) or 
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price crosses above R1 with volume spike and above EMA50
            if price > r1_12h[i] and vol > 1.3 * vol_ma and price > ema_50_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 with volume spike and below EMA50
            elif price < s1_12h[i] and vol > 1.3 * vol_ma and price < ema_50_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point or breaks below S1 (invalidates support)
            if price < pivot_12h[i] or price < s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point or breaks above R1 (invalidates resistance)
            if price > pivot_12h[i] or price > r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_S1R1_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0