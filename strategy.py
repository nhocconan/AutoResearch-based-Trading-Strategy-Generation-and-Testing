#!/usr/bin/env python3
name = "1d_WeeklyPivot_BullBear_Mode_v2"
timeframe = "1d"
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
    
    # Get weekly data
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using previous week)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Pivot point: (H + L + C)/3
    pp = (high_w + low_w + close_w) / 3
    # Resistance 1: 2*P - L
    r1 = 2 * pp - low_w
    # Support 1: 2*P - H
    s1 = 2 * pp - high_w
    
    # Align to daily
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    
    # Daily trend filter: 20-day EMA
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Volume filter: today's volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for EMA20
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(ema20[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above S1 and below PP (bullish bounce zone) + above EMA20 + volume
            if (close[i] > s1_aligned[i] and close[i] < pp_aligned[i] and
                close[i] > ema20[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below R1 and above PP (bearish rejection zone) + below EMA20 + volume
            elif (close[i] < r1_aligned[i] and close[i] > pp_aligned[i] and
                  close[i] < ema20[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses above PP (breakout) or below S1 (breakdown)
            if close[i] >= pp_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses below PP (breakdown) or above R1 (breakout)
            if close[i] <= pp_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals