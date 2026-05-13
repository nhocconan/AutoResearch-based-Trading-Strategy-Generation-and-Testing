#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate session hours once
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4H data ONCE for trend (Camarilla R1/S1 levels)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla Pivot levels on 4H
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4H bar's OHLC for Camarilla calculation
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Calculate pivot and ranges
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla levels
    r1_4h = close_4h + (range_4h * 1.1 / 12)
    s1_4h = close_4h - (range_4h * 1.1 / 12)
    r2_4h = close_4h + (range_4h * 1.1 / 6)
    s2_4h = close_4h - (range_4h * 1.1 / 6)
    
    # Align Camarilla levels to 1H
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    
    # Load 1D data ONCE for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1D average volume
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(r2_4h_aligned[i]) or np.isnan(s2_4h_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1H volume > 1.5x 20-day average 1D volume
        vol_filter = volume[i] > (1.5 * vol_avg_1d_aligned[i])
        
        if position == 0:
            # LONG: Price breaks above R1 with volume, target R2
            if (close[i] > r1_4h_aligned[i]) and vol_filter:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with volume, target S2
            elif (close[i] < s1_4h_aligned[i]) and vol_filter:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R2 or breaks below S1 (reversal)
            if (close[i] >= r2_4h_aligned[i]) or (close[i] < s1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reaches S2 or breaks above R1 (reversal)
            if (close[i] <= s2_4h_aligned[i]) or (close[i] > r1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals