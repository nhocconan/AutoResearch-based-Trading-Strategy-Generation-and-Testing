#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and ADX trend filter.
# Long when price touches Camarilla S1 or S2 AND 1d volume > 1.3x 20-period average AND ADX(14) > 25 (trending market).
# Short when price touches Camarilla R1 or R2 AND 1d volume > 1.3x 20-period average AND ADX(14) > 25.
# Exit when price reaches opposite Camarilla level (S1/R1 for long, R2/S2 for short) or closes beyond entry level.
# Uses 12h timeframe as specified, with 1d Camarilla levels, volume and ADX for higher timeframe context.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_Camarilla_Pivot_Reversal_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Camarilla, volume and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous day
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate pivot and ranges
    pivot = (high_d + low_d + close_d) / 3
    range_val = high_d - low_d
    
    # Camarilla levels
    r1 = close_d + range_val * 1.1 / 12
    r2 = close_d + range_val * 1.1 / 6
    s1 = close_d - range_val * 1.1 / 12
    s2 = close_d - range_val * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe (previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_d, s2)
    
    # Daily volume filter: current volume > 1.3x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.3 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Daily ADX(14) for trend strength
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # Directional Movement
    plus_dm = np.where((high_d - np.roll(high_d, 1)) > (np.roll(low_d, 1) - low_d), 
                       np.maximum(high_d - np.roll(high_d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_d, 1) - low_d) > (high_d - np.roll(high_d, 1)), 
                        np.maximum(np.roll(low_d, 1) - low_d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Trend filter: ADX > 25
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches S1 or S2, volume filter, trending market
            long_cond = ((low[i] <= s1_aligned[i]) or (low[i] <= s2_aligned[i])) and volume_filter[i] and trend_filter[i]
            # Short conditions: price touches R1 or R2, volume filter, trending market
            short_cond = ((high[i] >= r1_aligned[i]) or (high[i] >= r2_aligned[i])) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R1 or closes below S1/S2
            if (high[i] >= r1_aligned[i]) or (close[i] < s1_aligned[i]) or (close[i] < s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or closes above R1/R2
            if (low[i] <= s1_aligned[i]) or (close[i] > r1_aligned[i]) or (close[i] > r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals