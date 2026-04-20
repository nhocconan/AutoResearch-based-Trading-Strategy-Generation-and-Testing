#!/usr/bin/env python3
# 1h_4d_Camarilla_R1S1_Breakout_VolumeTrend
# Hypothesis: On 1h timeframe, trade breakouts at 4h Camarilla R1/S1 levels with volume confirmation.
# Uses 1d ADX to filter regime: ADX < 25 (range) = fade at R1/S1, ADX > 25 (trend) = breakout at R1/S1.
# Targets 15-35 trades/year by requiring confluence of level, volume, and regime filter.
# Works in bull/bear: range fading works in sideways/consolidation, trend breakout works in strong moves.

name = "1h_4d_Camarilla_R1S1_Breakout_VolumeTrend"
timeframe = "1h"
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
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price for pivot calculation
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    
    # Pivot point and ranges
    pivot_4h = typical_price_4h
    range_4h = high_4h - low_4h
    
    # Camarilla levels: R1, S1
    r1_4h = close_4h + (range_4h * 1.1 / 12)
    s1_4h = close_4h - (range_4h * 1.1 / 12)
    
    # Align 4h levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Calculate 1d ADX for trend/ranging filter (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Smoothed TR and DM
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
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Ranging market (ADX < 25): fade at R1/S1
            if adx_aligned[i] < 25:
                # Long near S1 with volume confirmation
                if (close[i] <= s1_aligned[i] * 1.005 and 
                    close[i] >= s1_aligned[i] * 0.995 and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.20
                    position = 1
                # Short near R1 with volume confirmation
                elif (close[i] >= r1_aligned[i] * 0.995 and 
                      close[i] <= r1_aligned[i] * 1.005 and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.20
                    position = -1
            # Trending market (ADX > 25): breakout at R1/S1
            elif adx_aligned[i] > 25:
                # Long breakout above R1 with volume
                if (close[i] > r1_aligned[i] * 1.005 and 
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.20
                    position = 1
                # Short breakdown below S1 with volume
                elif (close[i] < s1_aligned[i] * 0.995 and 
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] >= r1_aligned[i] * 0.995) or \
               (adx_aligned[i] > 25 and close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] <= s1_aligned[i] * 1.005) or \
               (adx_aligned[i] > 25 and close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals