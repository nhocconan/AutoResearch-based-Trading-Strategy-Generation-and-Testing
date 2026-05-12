#!/usr/bin/env python3
name = "12h_Weekly_Pivot_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Calculation (R1, S1) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = 2 * pivot - low_1w
    weekly_s1 = 2 * pivot - high_1w
    
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === Weekly Trend Filter: EMA50 on weekly close ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Weekly Volume Spike Filter ===
    vol_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = vol_1w > (2.5 * vol_avg_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure weekly EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above weekly R1 + above weekly EMA50 + volume spike
            if (close[i] > weekly_r1_aligned[i] and
                close[i] > ema50_1w_aligned[i] and
                vol_spike_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly S1 + below weekly EMA50 + volume spike
            elif (close[i] < weekly_s1_aligned[i] and
                  close[i] < ema50_1w_aligned[i] and
                  vol_spike_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly S1 or below weekly EMA50
            if close[i] < weekly_s1_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly R1 or above weekly EMA50
            if close[i] > weekly_r1_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals