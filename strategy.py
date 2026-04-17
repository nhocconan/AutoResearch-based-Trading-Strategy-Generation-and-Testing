#!/usr/bin/env python3
"""
Hypothesis: 4h Bollinger Band breakout with volume confirmation and ATR-based trend filter.
Long when price closes above upper BB AND volume > 1.5x average AND ATR(14) > ATR(50) (trending).
Short when price closes below lower BB AND volume > 1.5x average AND ATR(14) > ATR(50).
Exit when price reverts to middle BB OR ATR(14) < ATR(50) (ranging).
Uses 4h for BB and volume, 1d for ATR regime filter to avoid whipsaw in sideways markets.
Target: 75-200 total trades over 4 years (19-50/year). BB breakouts capture momentum,
volume confirmation filters fakeouts, ATR regime ensures trades only in trending conditions.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
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
    
    # Get 4h data for Bollinger Bands and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Bollinger Bands on 4h timeframe (20-period, 2 std)
    close_4h_series = pd.Series(close_4h)
    bb_middle = close_4h_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_4h_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d timeframe
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and ATR(50) for trend regime
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align 4h Bollinger Bands to 4h timeframe (no alignment needed)
    bb_upper_aligned = bb_upper
    bb_lower_aligned = bb_lower
    bb_middle_aligned = bb_middle
    
    # Align 1d ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        bu = bb_upper_aligned[i]
        bl = bb_lower_aligned[i]
        bm = bb_middle_aligned[i]
        atr14 = atr_14_aligned[i]
        atr50 = atr_50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Trend condition: ATR(14) > ATR(50) indicates trending market
        is_trending = atr14 > atr50
        
        if position == 0:
            # Long: price > upper BB AND volume > 1.5x avg AND trending
            if price > bu and vol > 1.5 * vol_ma and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price < lower BB AND volume > 1.5x avg AND trending
            elif price < bl and vol > 1.5 * vol_ma and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < middle BB OR not trending
            if price < bm or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > middle BB OR not trending
            if price > bm or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerBreakout_Volume_ATRTrend"
timeframe = "4h"
leverage = 1.0