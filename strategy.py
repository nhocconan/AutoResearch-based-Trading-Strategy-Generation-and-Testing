#!/usr/bin/env python3
name = "12h_WeeklyPivot_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # === Weekly Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C)/3
    pivot = (high_1w + low_1w + close_1w) / 3
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * pivot - low_1w
    weekly_s1 = 2 * pivot - high_1w
    
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === Weekly EMA34 Trend Filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily Volume Spike Filter ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above weekly R1 + above weekly EMA34 + volume spike
            if (close[i] > weekly_r1_aligned[i] and
                close[i] > ema34_1w_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly S1 + below weekly EMA34 + volume spike
            elif (close[i] < weekly_s1_aligned[i] and
                  close[i] < ema34_1w_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly S1 or below weekly EMA34
            if close[i] < weekly_s1_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly R1 or above weekly EMA34
            if close[i] > weekly_r1_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals