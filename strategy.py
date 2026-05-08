#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_S2_S4_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly trend: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly pivot points (S2, S4, R2, R4)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    H_prev = np.roll(high_1w, 1)
    L_prev = np.roll(low_1w, 1)
    C_prev = np.roll(close_1w, 1)
    H_prev[0] = np.nan
    L_prev[0] = np.nan
    C_prev[0] = np.nan
    
    pivot = (H_prev + L_prev + C_prev) / 3
    range_hl = H_prev - L_prev
    
    # S2 = pivot - range
    S2 = pivot - range_hl
    # S4 = pivot - 2 * range
    S4 = pivot - 2 * range_hl
    # R2 = pivot + range
    R2 = pivot + range_hl
    # R4 = pivot + 2 * range
    R4 = pivot + 2 * range_hl
    
    # Align to daily timeframe
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(S2_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R2 + weekly uptrend + volume spike
            long_cond = (close[i] > R2_aligned[i]) and \
                        (close[i] > ema_50_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: break below S2 + weekly downtrend + volume spike
            short_cond = (close[i] < S2_aligned[i]) and \
                         (close[i] < ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below weekly pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above weekly pivot (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals