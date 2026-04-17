#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 pivot breakout with volume confirmation and 1w ADX trend filter.
Long when price breaks above R3 AND volume > 1.8x average AND weekly ADX > 25 (trending).
Short when price breaks below S3 AND volume > 1.8x average AND weekly ADX > 25.
Exit when price reverts to H4/L4 level OR weekly ADX < 20 (range market).
Uses 1d for price/volume, 1w for ADX filter to avoid whipsaw in ranging markets.
Target: 30-100 total trades over 4 years (7-25/year). Camarilla pivots provide precise intraday levels,
volume confirmation reduces fakeouts, weekly ADX ensures we only trade in strong trends.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels on 1d timeframe
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # H4 and L4 (midpoints between R3/R4 and S3/S4)
    h4 = (r3 + r4) / 2.0
    l4 = (s3 + s4) / 2.0
    
    # Calculate volume average (20-period) on 1d
    volume_series = pd.Series(volume_1d)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w timeframe (14-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
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
    
    # Align 1d Camarilla levels, volume MA, and 1w ADX to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R3 AND volume > 1.8x avg AND weekly ADX > 25 (trending)
            if price > r3 and vol > 1.8 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < S3 AND volume > 1.8x avg AND weekly ADX > 25 (trending)
            elif price < s3 and vol > 1.8 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < H4 OR weekly ADX < 20 (range market)
            if price < h4 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > L4 OR weekly ADX < 20 (range market)
            if price > l4 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_CamarillaR3S3_Volume_1wADX_Filter"
timeframe = "1d"
leverage = 1.0