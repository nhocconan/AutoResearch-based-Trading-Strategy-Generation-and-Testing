#!/usr/bin/env python3
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
    
    # Load 1w data for weekly pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close for weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels (P, R1, S1, R2, S2)
    pivot_w = (high_1w + low_1w + close_1w) / 3
    range_w = high_1w - low_1w
    r1_w = pivot_w + range_w * 1
    s1_w = pivot_w - range_w * 1
    r2_w = pivot_w + range_w * 2
    s2_w = pivot_w - range_w * 2
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA100 for trend filter
    ema_100_1d = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values  # Note: using weekly close for EMA, but aligned to daily
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Align 1d EMA100 to 6h timeframe
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready or outside session
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or
            np.isnan(r2_w_aligned[i]) or np.isnan(s2_w_aligned[i]) or
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above weekly R2 with volume AND above 1d EMA100 (uptrend)
            if (close[i] > r2_w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i] and 
                close[i] > ema_100_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below weekly S2 with volume AND below 1d EMA100 (downtrend)
            elif (close[i] < s2_w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i] and 
                  close[i] < ema_100_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to weekly pivot level
            if position == 1:
                if not np.isnan(pivot_w_aligned[i]) and close[i] < pivot_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not np.isnan(pivot_w_aligned[i]) and close[i] > pivot_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyPivot_R2_S2_Breakout_1dEMA100_Trend_Volume_Session"
timeframe = "6h"
leverage = 1.0