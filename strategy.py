#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot R1/S1 Breakout with Volume Spike and Weekly Trend Filter.
Long when price breaks above R1 with volume > 2.0x average and weekly close > weekly open (bullish weekly candle).
Short when price breaks below S1 with volume > 2.0x average and weekly close < weekly open (bearish weekly candle).
Exit when price reverts to pivot point (PP).
Uses 1d for Camarilla pivot calculation, 6h for price/volume, 1w for trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1, PP)
    def calculate_camarilla(high, low, close):
        pp = (high + low + close) / 3.0
        r1 = close + (high - low) * 1.1 / 12.0
        s1 = close - (high - low) * 1.1 / 12.0
        return pp, r1, s1
    
    pp_1d = np.zeros_like(close_1d)
    r1_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        pp, r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pp_1d[i] = pp
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Calculate 1w trend: bullish if close > open, bearish if close < open
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align 1d indicators to 6h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Align 1w indicators to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))  # align to 1d first
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish_aligned)  # then to 6h (1d->6h)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish_aligned)
    
    # Calculate volume spike (current volume > 2.0x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        pp = pp_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and bullish weekly candle
            if price > r1 and vol_spike and weekly_bull:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and bearish weekly candle
            elif price < s1 and vol_spike and weekly_bear:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point
            if price <= pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point
            if price >= pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_VolumeSpike_WeeklyTrend"
timeframe = "6h"
leverage = 1.0