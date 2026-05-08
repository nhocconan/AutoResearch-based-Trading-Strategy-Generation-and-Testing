#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot levels (R1, S1) from previous week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot using previous week's data
    H_prev = np.roll(high_1w, 1)
    L_prev = np.roll(low_1w, 1)
    C_prev = np.roll(close_1w, 1)
    H_prev[0] = np.nan
    L_prev[0] = np.nan
    C_prev[0] = np.nan
    
    pivot = (H_prev + L_prev + C_prev) / 3
    range_hl = H_prev - L_prev
    
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # Align weekly levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Weekly trend: EMA34 on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 + weekly uptrend + volume spike
            long_cond = (close[i] > R1_aligned[i]) and \
                        (close[i] > ema_34_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: break below S1 + weekly downtrend + volume spike
            short_cond = (close[i] < S1_aligned[i]) and \
                         (close[i] < ema_34_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below pivot (mean reversion to mean)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above pivot (mean reversion to mean)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals