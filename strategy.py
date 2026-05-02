#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power with 1d regime filter
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via smoothed medians.
# Elder Ray Power (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
# 1d ADX > 25 filters for trending markets to avoid chop.
# Only take Long when: Alligator aligned bullish (JAW > TEETH > LIPS) AND Bull Power > 0 AND 1d ADX > 25
# Only take Short when: Alligator aligned bearish (JAW < TEETH < LIPS) AND Bear Power > 0 AND 1d ADX > 25
# Exit when Alligator alignment fails OR Elder Ray power turns negative.
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Works in bull markets (catching strong uptrends) and bear markets (catching strong downtrends)
# by requiring both Alligator alignment and Elder Ray confirmation with ADX regime filter.

name = "6h_WilliamsAlligator_ElderRay_1dADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Williams Alligator on 6h data (using medians)
    # JAW: Smoothed Median (13 periods, 8 offset)
    # TEETH: Smoothed Median (8 periods, 5 offset)
    # LIPS: Smoothed Median (5 periods, 3 offset)
    median = (high + low) / 2
    
    def smoothed_median(values, period, offset):
        sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
        return pd.Series(sma).rolling(window=period, min_periods=period).mean().shift(offset).values
    
    jaw = smoothed_median(median, 13, 8)
    teeth = smoothed_median(median, 8, 5)
    lips = smoothed_median(median, 5, 3)
    
    # Calculate Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (14-period)
    def smooth_values(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr14 = smooth_values(tr, 14)
    plus_dm14 = smooth_values(plus_dm, 14)
    minus_dm14 = smooth_values(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = np.zeros_like(dx)
    adx[27] = np.nanmean(dx[14:28])  # first ADX value
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align Alligator and Elder Ray to 6h (they're already on 6h, but ensure alignment for consistency)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish AND Bull Power positive AND ADX > 25
            if (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                bull_power_aligned[i] > 0 and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish AND Bear Power positive AND ADX > 25
            elif (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and
                  bear_power_aligned[i] > 0 and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment fails OR Bull Power turns negative
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment fails OR Bear Power turns negative
            if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or bear_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals