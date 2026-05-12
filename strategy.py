#!/usr/bin/env python3
name = "1d_WeeklyPivotBreakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w Data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === Calculate weekly pivot points (PP, R1, S1) from previous week ===
    # Previous week's high, low, close
    prev_week_high = high_1w
    prev_week_low = low_1w
    prev_week_close = close_1w
    
    # Weekly pivot calculation
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    
    # === Calculate weekly EMA34 for trend filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Volume spike detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align weekly data to daily timeframe (previous week's levels available at open)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + above weekly EMA34 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + below weekly EMA34 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S1 (reversal) or below weekly EMA34
            if close[i] < s1_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R1 (reversal) or above weekly EMA34
            if close[i] > r1_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals