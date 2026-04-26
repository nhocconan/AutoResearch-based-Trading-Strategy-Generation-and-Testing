#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above R1 with bullish 12h trend and volume spike.
Enters short when price breaks below S1 with bearish 12h trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-200 total trades over 4 years.
Works in both bull and bear markets by following the 12h trend direction only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate prior day's Camarilla levels (using prior daily bar)
    # We'll use prior completed daily bar to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC (shifted by 1 to use completed daily bar)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close[0] = np.nan  # First bar has no prior
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Camarilla calculations
    rang = prior_high - prior_low
    camarilla_r1 = prior_close + rang * 1.1 / 12
    camarilla_s1 = prior_close - rang * 1.1 / 12
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    camarilla_r4 = prior_close + rang * 1.1 / 2
    camarilla_s4 = prior_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 trend
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need prior daily bar + EMA34)
    start_idx = 34 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 + bullish 12h trend + volume spike
        if close[i] > r1_4h[i] and close[i] > ema_34_12h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S1 + bearish 12h trend + volume spike
        elif close[i] < s1_4h[i] and close[i] < ema_34_12h_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s1_4h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_4h[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0