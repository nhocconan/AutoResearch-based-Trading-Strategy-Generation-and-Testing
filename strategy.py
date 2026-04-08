#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_squeeze_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume squeeze filter.
# Long when price breaks above H3 with volume > 1.5x average, short when breaks below L3.
# Uses 1d ADX < 25 to filter ranging markets and avoid false breakouts.
# Designed for 15-30 trades/year on 12h to minimize fee drag. Works in bull/bear via volatility filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_squeeze_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    
    # Support levels
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # H3 and L3 are the key levels for breakouts
    h3_1d = r3_1d
    l3_1d = s3_1d
    
    # Align H3 and L3 to 12h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 1d ADX for trend filter
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (similar to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_period = 14
    atr_1d = wilders_smooth(tr, atr_period)
    plus_di_1d = wilders_smooth(plus_dm, atr_period) / atr_1d * 100
    minus_di_1d = wilders_smooth(minus_dm, atr_period) / atr_1d * 100
    dx_1d = np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d) * 100
    adx_1d = wilders_smooth(dx_1d, atr_period)
    
    # Align ADX to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(30, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # ADX filter: only trade when ADX < 25 (ranging market) to avoid false breakouts in strong trends
        adx_filter = adx_1d_aligned[i] < 25
        
        if position == 1:  # Long position
            # Exit: price closes below L3 or volume dries up
            if close[i] < l3_1d_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 or volume dries up
            if close[i] > h3_1d_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above H3 with volume and ADX filter
            if (close[i] > h3_1d_aligned[i] and 
                volume_filter and 
                adx_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume and ADX filter
            elif (close[i] < l3_1d_aligned[i] and 
                  volume_filter and 
                  adx_filter):
                position = -1
                signals[i] = -0.25
    
    return signals