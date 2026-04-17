#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.8x average AND daily ADX > 20 (trending).
Short when price breaks below Camarilla S1 AND volume > 1.8x average AND daily ADX > 20.
Exit when price reverts to Camarilla midpoint (M1/M2) OR daily ADX < 15 (range market).
Uses 12h for price/volume, 1d for ADX filter to avoid whipsaw in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide precise intraday support/resistance,
volume confirmation reduces fakeouts, daily ADX ensures we only trade in strong trends.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
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
    
    # Get 12h data for Camarilla levels and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla pivot levels on 12h timeframe (previous day)
    # Pivot = (high_prev + low_prev + close_prev) / 3
    # R1 = close + (high_prev - low_prev) * 1.1 / 12
    # S1 = close - (high_prev - low_prev) * 1.1 / 12
    # M1 = close + (high_prev - low_prev) * 1.1 / 6
    # M2 = close - (high_prev - low_prev) * 1.1 / 6
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    close_series = pd.Series(close_12h)
    
    # Shift by 1 to use previous bar's data (completed 12h bar)
    high_prev = high_series.shift(1)
    low_prev = low_series.shift(1)
    close_prev = close_series.shift(1)
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    camarilla_r1 = close_prev + range_prev * 1.1 / 12
    camarilla_s1 = close_prev - range_prev * 1.1 / 12
    camarilla_m1 = close_prev + range_prev * 1.1 / 6  # Upper midpoint
    camarilla_m2 = close_prev - range_prev * 1.1 / 6  # Lower midpoint
    camarilla_mid = (camarilla_m1 + camarilla_m2) / 2  # Overall midpoint
    
    # Calculate volume average (20-period) on 12h
    volume_series = pd.Series(volume_12h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr != 0, atr, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr != 0, atr, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h Camarilla levels, volume MA, and 1d ADX to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        mid = camarilla_mid_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.8x avg AND daily ADX > 20 (trending)
            if price > r1 and vol > 1.8 * vol_ma and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.8x avg AND daily ADX > 20 (trending)
            elif price < s1 and vol > 1.8 * vol_ma and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla midpoint OR daily ADX < 15 (range market)
            if price < mid or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla midpoint OR daily ADX < 15 (range market)
            if price > mid or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1dADX_Filter"
timeframe = "12h"
leverage = 1.0