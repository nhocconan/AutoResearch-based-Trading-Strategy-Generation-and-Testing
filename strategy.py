#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when: price breaks above Camarilla R3 + 1d EMA34 up + volume > 1.5x avg volume.
Short when: price breaks below Camarilla S3 + 1d EMA34 down + volume > 1.5x avg volume.
Exit when: price reverts to Camarilla H5/L5 or trend reverses.
Designed for 12h timeframe with discrete 0.25 position size to limit fee drag (~25 trades/year).
Works in both bull and bear via trend filter and volatility-based entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar (use shift to avoid look-ahead)
    # Camarilla: based on previous bar's range
    prev_close = pd.Series(close).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    
    # Camarilla R3, S3, H5, L5
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # H5 = close + (high - low) * 1.1/2
    # L5 = close - (high - low) * 1.1/2
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + rng * 1.1 / 4
    camarilla_s3 = prev_close - rng * 1.1 / 4
    camarilla_h5 = prev_close + rng * 1.1 / 2
    camarilla_l5 = prev_close - rng * 1.1 / 2
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # 1d EMA34 trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d EMA34 slope (trend direction)
    ema_slope = np.diff(ema_34_1d_aligned, prepend=ema_34_1d_aligned[0])
    ema_up = ema_slope > 0
    ema_down = ema_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    fixed_size = 0.25
    
    # Warmup: need 20 for volume MA, 1 for Camarilla (uses prev bar)
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_h5[i]) or np.isnan(camarilla_l5[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for breakout with volume spike and trend alignment
            long_breakout = close_val > camarilla_r3[i]
            short_breakout = close_val < camarilla_s3[i]
            
            if long_breakout and volume_spike[i] and ema_up[i]:
                signals[i] = size
                position = 1
            elif short_breakout and volume_spike[i] and ema_down[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to H5/L5 or trend reverses
            if close_val < camarilla_h5[i] or not ema_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to H5/L5 or trend reverses
            if close_val > camarilla_l5[i] or not ema_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0