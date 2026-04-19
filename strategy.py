#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_1d_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6-period ATR for Donchian filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    
    # For weekly data, calculate pivot from previous week
    pp_1w = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3
    r1_1w = 2 * pp_1w - low_1w[:-1]
    s1_1w = 2 * pp_1w - high_1w[:-1]
    r2_1w = pp_1w + (high_1w[:-1] - low_1w[:-1])
    s2_1w = pp_1w - (high_1w[:-1] - low_1w[:-1])
    r3_1w = high_1w[:-1] + 2 * (pp_1w - low_1w[:-1])
    s3_1w = low_1w[:-1] - 2 * (high_1w[:-1] - pp_1w)
    
    # Align weekly pivot levels to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate daily pivot points (using previous day's OHLC)
    pp_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    r1_1d = 2 * pp_1d - low_1d[:-1]
    s1_1d = 2 * pp_1d - high_1d[:-1]
    r2_1d = pp_1d + (high_1d[:-1] - low_1d[:-1])
    s2_1d = pp_1d - (high_1d[:-1] - low_1d[:-1])
    r3_1d = high_1d[:-1] + 2 * (pp_1d - low_1d[:-1])
    s3_1d = low_1d[:-1] - 2 * (high_1d[:-1] - pp_1d)
    
    # Align daily pivot levels to 6h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(pp_1d_aligned[i]) or \
           np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        # Determine weekly pivot direction: bullish if price > weekly pivot
        weekly_bullish = price > pp_1w_aligned[i]
        weekly_bearish = price < pp_1w_aligned[i]
        
        # Determine daily pivot direction: bullish if price > daily pivot
        daily_bullish = price > pp_1d_aligned[i]
        daily_bearish = price < pp_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + weekly/daily bullish + volume
            if (price > donchian_high[i] and 
                weekly_bullish and daily_bullish and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + weekly/daily bearish + volume
            elif (price < donchian_low[i] and 
                  weekly_bearish and daily_bearish and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to Donchian midpoint or weekly/daily turns bearish
            if (price < donchian_mid[i] or 
                not weekly_bullish or not daily_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to Donchian midpoint or weekly/daily turns bullish
            if (price > donchian_mid[i] or 
                not weekly_bearish or not daily_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals