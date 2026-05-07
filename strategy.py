#!/usr/bin/env python3
"""
6H_WeeklyPivot_DailyTrend_VolumeBreakout
Hypothesis: On 6h timeframe, use weekly pivot points for structure and daily trend (EMA34) for bias.
Long when: price > daily EMA34 AND 6h close breaks above weekly R1 with volume spike.
Short when: price < daily EMA34 AND 6h close breaks below weekly S1 with volume spike.
Weekly pivots provide institutional levels; daily EMA34 filters trend direction; volume confirms breakout strength.
Designed for 50-150 total trades over 4 years (12-37/year) with size 0.25.
Works in bull (breakouts with trend) and bear (fades at weekly S1/R1 in downtrend).
"""
name = "6H_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Get daily data for trend filter (EMA34)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Using previous week's OHLC to avoid look-ahead
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_w + low_w + close_w) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pp - low_w
    s1 = 2 * pp - high_w
    # R2 = P + (H - L), S2 = P - (H - L) - for potential exits
    r2 = pp + (high_w - low_w)
    s2 = pp - (high_w - low_w)
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    
    # Calculate daily EMA34 for trend filter
    close_d = df_d['close'].values
    ema_34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_d_aligned = align_htf_to_ltf(prices, df_d, ema_34_d)
    
    # Volume filter: current 6h volume > 2.0 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient warmup for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily EMA34, 6h close breaks above weekly R1, volume spike
            if (close[i] > ema_34_d_aligned[i] and 
                close[i] > r1_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34, 6h close breaks below weekly S1, volume spike
            elif (close[i] < ema_34_d_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly R2 (strong resistance) or daily EMA34
            if close[i] < ema_34_d_aligned[i] or close[i] < r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly S2 (strong support) or daily EMA34
            if close[i] > ema_34_d_aligned[i] or close[i] > s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals