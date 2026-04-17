#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume spike and ADX trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.5x 20-period average AND 1d ADX > 25.
Short when price breaks below Camarilla S1 AND volume > 1.5x 20-period average AND 1d ADX > 25.
Exit when price reverts to Camarilla midpoint (PP).
Uses 12h for price/volume/Camarilla, 1d for ADX trend filter to avoid whipsaw in ranging markets.
Targets 50-150 total trades over 4 years (12-37/year). Camarilla levels provide high-probability breakout points,
volume confirmation reduces fakeouts, ADX ensures we trade only in trending markets.
Works in bull markets (captures uptrends with rising ADX) and bear markets (captures downtrends with rising ADX).
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla pivot levels (R1, S1, PP) using previous day's OHLC
    # We need to shift by 1 to use completed 12h bar for pivot calculation
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_point = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot_point + (range_hl * 1.1 / 12)
    s1 = pivot_point - (range_hl * 1.1 / 12)
    
    # Calculate 12h volume average (20-period)
    volume_series = pd.Series(volume_12h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.insert(tr1, 0, 0)
    atr = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h Camarilla levels, volume MA, and 1d ADX to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    pivot_point_aligned = align_htf_to_ltf(prices, df_12h, pivot_point)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_point_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pivot_point_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.5x avg AND 1d ADX > 25 (trending market)
            if price > r1_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.5x avg AND 1d ADX > 25 (trending market)
            elif price < s1_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla pivot point (PP)
            if price < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla pivot point (PP)
            if price > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0