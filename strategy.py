#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume spike. Uses discrete 0.30 position size.
Long when: price breaks above R1 + 12h EMA50 uptrend + volume > 2.0 * 20-bar avg volume.
Short when: price breaks below S1 + 12h EMA50 downtrend + volume > 2.0 * 20-bar avg volume.
Exit when: price crosses Camarilla midpoint (PP) or touches opposite level (S1 for long, R1 for short).
Designed for low trade frequency (target 20-40/year) to minimize fee drag and work in both bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1, S1, PP (pivot point)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 4h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 50 for 12h EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.30  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R1 + 12h EMA50 uptrend + volume spike
            long_entry = (close_val > camarilla_r1_aligned[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below S1 + 12h EMA50 downtrend + volume spike
            short_entry = (close_val < camarilla_s1_aligned[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to PP or touches S1 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R1 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0