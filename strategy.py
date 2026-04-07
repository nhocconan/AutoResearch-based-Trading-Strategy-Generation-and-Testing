#!/usr/bin/env python3
"""
6H 1W Pivot Breakout with 1D Trend Filter
Buy when price breaks above weekly R4 with expanding volume and daily EMA rising
Sell when price breaks below weekly S4 with expanding volume and daily EMA falling
Exit when price crosses back to weekly pivot point
Uses weekly pivot levels as dynamic support/resistance - works in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_pivot_breakout_1d_trend_filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly pivot points from weekly data ===
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    weekly_r4 = weekly_r3 + (weekly_high - weekly_low)
    weekly_s4 = weekly_s3 - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # === Daily EMA trend filter ===
    df_daily = get_htf_data(prices, '1d')
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema_daily_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below weekly pivot
            if close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above weekly pivot
            if close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Weekly R4/S4 breakout with volume confirmation AND daily EMA trend
            if close[i] > r4_aligned[i] and ema_daily_aligned[i] > ema_daily_aligned[i-1]:
                # Breakout above weekly R4 with rising daily EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_aligned[i] and ema_daily_aligned[i] < ema_daily_aligned[i-1]:
                # Breakdown below weekly S4 with falling daily EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals