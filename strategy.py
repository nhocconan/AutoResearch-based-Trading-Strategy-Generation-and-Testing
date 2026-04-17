#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 level AND 1w EMA34 is rising AND 1d volume > 2.0x 20-day average volume.
Short when price breaks below Camarilla S1 level AND 1w EMA34 is falling AND 1d volume > 2.0x 20-day average volume.
Exit when price returns to the Camarilla pivot point (mean reversion) or opposite breakout occurs.
Uses 1w for trend direction, 1d for Camarilla levels and volume filters. Designed to capture strong intraday trends
with volume confirmation while avoiding choppy markets. Target: 15-25 trades/year per symbol.
"""

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
    
    # Get 1d data for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels on 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1 / 12)
    # S1 = C - (Range * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)
    
    # Calculate 20-day average volume on 1d for volume spike filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA34 and its slope (rising/falling)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_prev = np.roll(ema_34_1w, 1)
    ema_34_1w_prev[0] = np.nan
    ema_rising = ema_34_1w > ema_34_1w_prev
    ema_falling = ema_34_1w < ema_34_1w_prev
    
    # Align all 1d indicators to 1d timeframe (since primary timeframe is 1d, no alignment needed)
    # But we still use align_htf_to_ltf for consistency and to handle any potential index issues
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Align 1w EMA34 and its slope to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current 1d volume > 2.0x 20-day average
        # Get the aligned 1d volume for this timestamp
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_spike = not np.isnan(volume_1d_aligned[i]) and \
                      not np.isnan(vol_ma_20_1d_aligned[i]) and \
                      volume_1d_aligned[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_1d_aligned[i]
        breakout_down = close[i] < s1_1d_aligned[i]
        
        # Mean reversion exit condition: price returns to pivot point
        revert_to_pivot = abs(close[i] - pivot_1d_aligned[i]) < 0.001 * close[i]  # within 0.1%
        
        # Opposite breakout exit condition
        opposite_breakout = (position == 1 and breakout_down) or (position == -1 and breakout_up)
        
        if position == 0:
            # Long: breakout above R1 with rising 1w EMA34 and volume spike
            if (breakout_up and ema_rising_aligned[i] == 1.0 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 with falling 1w EMA34 and volume spike
            elif (breakout_down and ema_falling_aligned[i] == 1.0 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point OR opposite breakout
            if (revert_to_pivot or opposite_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR opposite breakout
            if (revert_to_pivot or opposite_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0