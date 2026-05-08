#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_DailyTrend_Volume"
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
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly high/low/close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (standard formula: (H+L+C)/3)
    H_prev = np.roll(high_1w, 1)
    L_prev = np.roll(low_1w, 1)
    C_prev = np.roll(close_1w, 1)
    H_prev[0] = np.nan
    L_prev[0] = np.nan
    C_prev[0] = np.nan
    
    pivot_weekly = (H_prev + L_prev + C_prev) / 3
    range_weekly = H_prev - L_prev
    
    # Weekly resistance/support levels (similar to Camarilla but simpler)
    R1 = pivot_weekly + (range_weekly * 1.0)
    S1 = pivot_weekly - (range_weekly * 1.0)
    
    # Align weekly levels to 6h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_1w, pivot_weekly)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA34 for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (2x 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(pivot_weekly_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 + daily uptrend + volume spike
            long_cond = (close[i] > R1_aligned[i]) and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_spike[i]
            # Short: break below weekly S1 + daily downtrend + volume spike
            short_cond = (close[i] < S1_aligned[i]) and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below weekly pivot (mean reversion)
            if close[i] < pivot_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above weekly pivot (mean reversion)
            if close[i] > pivot_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals