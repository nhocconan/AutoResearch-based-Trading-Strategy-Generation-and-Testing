#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h Camarilla pivot structure + volume confirmation
# Donchian channels provide objective breakout levels that work in both trending and ranging markets.
# 12h Camarilla pivots (R3/S3, R4/S4) define institutional support/resistance for continuation vs reversal logic.
# Volume confirmation (1.5x 20-period EMA) filters low-conviction breakouts.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

name = "6h_Donchian20_12hCamarilla_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivots (R3, S3, R4, S4) from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True range for pivot calculation
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]  # first bar
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Camarilla levels
    r3_12h = pp_12h + (high_12h - low_12h) * 1.1 / 4.0
    s3_12h = pp_12h - (high_12h - low_12h) * 1.1 / 4.0
    r4_12h = pp_12h + (high_12h - low_12h) * 1.1 / 2.0
    s4_12h = pp_12h - (high_12h - low_12h) * 1.1 / 2.0
    
    # Align 12h Camarilla levels to 6h timeframe (wait for completed 12h bar)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate Donchian channels from previous 6h bar (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Donchian and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above R3 (bullish continuation zone)
            if close[i] > donchian_upper[i] and close[i] > r3_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below S3 (bearish continuation zone)
            elif close[i] < donchian_lower[i] and close[i] < s3_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower or below S3 (invalidates bullish structure)
            if close[i] < donchian_lower[i] or close[i] < s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper or above R3 (invalidates bearish structure)
            if close[i] > donchian_upper[i] or close[i] > r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals