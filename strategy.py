#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and 1w ADX trend filter.
Long when price breaks above R1 AND 12h volume > 1.5x 20-bar avg AND weekly ADX > 25.
Short when price breaks below S1 AND 12h volume > 1.5x 20-bar avg AND weekly ADX > 25.
Exit when price touches H4/L4 or opposite Camarilla level.
Uses 1w for ADX trend regime and 1d for Camarilla pivots, 12h for execution and volume.
Designed to capture institutional order flow at key pivot levels with volume confirmation.
Target: 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = np.maximum(high_1w - low_1w, 
                     np.absolute(high_1w - np.roll(close_1w, 1)),
                     np.absolute(low_1w - np.roll(close_1w, 1)))
    tr1[0] = high_1w[0] - low_1w[0]
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    # Smoothed TR, DM+-, DX
    tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    dm_plus_sum = pd.Series(dm_plus_14).rolling(window=14, min_periods=14).sum().values
    dm_minus_sum = pd.Series(dm_minus_14).rolling(window=14, min_periods=14).sum().values
    tr14_sum = pd.Series(tr14).rolling(window=14, min_periods=14).sum().values
    dx = 100 * np.abs(dm_plus_sum - dm_minus_sum) / (dm_plus_sum + dm_minus_sum + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + ((high-low)*1.1/2)
    # R3 = close + ((high-low)*1.1/4)
    # R2 = close + ((high-low)*1.1/6)
    # R1 = close + ((high-low)*1.1/12)
    # PP = (high+low+close)/3
    # S1 = close - ((high-low)*1.1/12)
    # S2 = close - ((high-low)*1.1/6)
    # S3 = close - ((high-low)*1.1/4)
    # S4 = close - ((high-low)*1.1/2)
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    r4 = close_1d + (range_1d * 1.1 / 2)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Get 12h data for execution and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Exit conditions: touch R4/S4 or opposite Camarilla level
        touch_r4 = abs(close[i] - r4_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_s4 = abs(close[i] - s4_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < s1_aligned[i]) or \
                         (position == -1 and close[i] > r1_aligned[i])
        
        if position == 0:
            # Long: break above R1 with volume confirmation and trend
            if (breakout_r1 and volume_confirmed and trending):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and trend
            elif (breakout_s1 and volume_confirmed and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch R4 or break below S1
            if (touch_r4 or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch S4 or break above R1
            if (touch_s4 or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_ADX_Trend"
timeframe = "12h"
leverage = 1.0