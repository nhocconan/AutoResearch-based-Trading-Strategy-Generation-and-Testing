#!/usr/bin/env python3
# 4h_1d_Camarilla_R1S1_Breakout_VolumeRegime_V1
# Hypothesis: On 4h timeframe, trade breakouts from 1d Camarilla R1/S1 levels with volume confirmation and ADX regime filter.
# In ranging markets (ADX < 25), trade reversals at R1/S1; in trending markets (ADX > 25), trade breakouts beyond R1/S1.
# Targets 20-40 trades/year by requiring confluence of level, volume, and regime filter.
# Works in both bull and bear markets due to adaptive regime filtering and volatility-adjusted position sizing.

name = "4h_1d_Camarilla_R1S1_Breakout_VolumeRegime_V1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1, R1, S2, R2
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    
    # Align 1d levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    
    # Calculate 1d ADX for trend/ranging filter (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM using Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Ranging market (ADX < 25): reverse at S1/R1
            if adx_aligned[i] < 25:
                # Long near S1 with volume confirmation
                if (close[i] <= s1_aligned[i] * 1.005 and 
                    close[i] >= s1_aligned[i] * 0.995 and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short near R1 with volume confirmation
                elif (close[i] >= r1_aligned[i] * 0.995 and 
                      close[i] <= r1_aligned[i] * 1.005 and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
            # Trending market (ADX > 25): breakout beyond S1/R1
            elif adx_aligned[i] > 25:
                # Long breakout above R1 with volume
                if (close[i] > r1_aligned[i] * 1.005 and 
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S1 with volume
                elif (close[i] < s1_aligned[i] * 0.995 and 
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] >= r1_aligned[i] * 0.995) or \
               (adx_aligned[i] > 25 and close[i] < s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] <= s1_aligned[i] * 1.005) or \
               (adx_aligned[i] > 25 and close[i] > r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals