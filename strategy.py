#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 level (bullish bias) AND 12h EMA50 is rising (trend filter) AND volume > 1.5x 20-period average volume. Enter short when price breaks below Camarilla S1 level AND 12h EMA50 is falling AND volume spike. Exit on trend reversal or price retracing to Camarilla midpoint. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 19-50 trades/year.
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
    
    # Get 12h data for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter (rising/falling)
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: rising if current > previous, falling if current < previous
    ema_50_12h_rising = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_12h_falling = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_12h_rising[1:] = ema_50_12h[1:] > ema_50_12h[:-1]
    ema_50_12h_falling[1:] = ema_50_12h[1:] < ema_50_12h[:-1]
    # Align to 4h timeframe
    ema_50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_rising)
    ema_50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_falling)
    
    # Calculate 12h Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_raw = df_12h['close'].values
    
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h_raw, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    camarilla_range = prev_high_12h - prev_low_12h
    r1 = prev_close_12h + 1.1 * camarilla_range / 12  # Camarilla R1
    s1 = prev_close_12h - 1.1 * camarilla_range / 12  # Camarilla S1
    mid = (r1 + s1) / 2  # Camarilla midpoint for exit
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    mid_aligned = align_htf_to_ltf(prices, df_12h, mid)
    
    # Volume confirmation: >1.5x 20-period average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 warmup (50) and volume MA warmup (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(mid_aligned[i]) or 
            np.isnan(ema_50_12h_rising_aligned[i]) or np.isnan(ema_50_12h_falling_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + 12h EMA50 rising
            long_signal = breakout_up and volume_spike[i] and ema_50_12h_rising_aligned[i]
            
            # Short: breakout below S1 + volume spike + 12h EMA50 falling
            short_signal = breakout_down and volume_spike[i] and ema_50_12h_falling_aligned[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend change (EMA50 falling) OR price retracing to Camarilla midpoint
            if ema_50_12h_falling_aligned[i] or close[i] < mid_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend change (EMA50 rising) OR price retracing to Camarilla midpoint
            if ema_50_12h_rising_aligned[i] or close[i] > mid_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0