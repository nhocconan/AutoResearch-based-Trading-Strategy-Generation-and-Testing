#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R3, S4)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = H - L
    range_1w = high_1w - low_1w
    # Resistance levels
    r3_1w = pivot_1w + (range_1w * 1.1)
    s4_1w = pivot_1w - (range_1w * 1.6)
    
    # Align weekly pivot levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation - 24-period average volume (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 + above daily EMA34 + volume confirmation
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 + below daily EMA34 + volume confirmation
            elif (close[i] < s4_1w_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below weekly pivot OR below daily EMA34
            weekly_pivot_1w = ((high_1w + low_1w + close_1w) / 3.0)
            weekly_pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
            if (close[i] < weekly_pivot_1w_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above weekly pivot OR above daily EMA34
            weekly_pivot_1w = ((high_1w + low_1w + close_1w) / 3.0)
            weekly_pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
            if (close[i] > weekly_pivot_1w_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals