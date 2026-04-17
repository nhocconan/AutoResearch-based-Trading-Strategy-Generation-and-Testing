#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + volume confirmation + weekly EMA200 trend filter.
Long when price breaks above Donchian upper band AND volume > 1.5x average AND close > weekly EMA200.
Short when price breaks below Donchian lower band AND volume > 1.5x average AND close < weekly EMA200.
Exit when price reverts to Donchian middle (20-period mean) OR weekly EMA200 slope flips.
Uses 1d for primary timeframe and 1w for HTF trend filter to capture major moves while avoiding counter-trend whipsaw.
Target: 30-100 total trades over 4 years (7-25/year). Donchian breakouts capture strong trends,
volume confirmation filters fakeouts, weekly EMA200 ensures alignment with major trend.
Works in bull markets (captures uptrends) and bear markets (captures downtrends) by only trading in direction of weekly trend.
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
    
    # Get 1d data for Donchian calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donchian_upper = high_1d_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1d_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Volume average (20-period) on 1d
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for weekly EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Align 1d indicators to 1d timeframe (no alignment needed)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    donchian_middle_aligned = donchian_middle
    volume_ma_aligned = volume_ma
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        ema200 = ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 1.5x avg AND price > weekly EMA200
            if price > du and vol > 1.5 * vol_ma and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 1.5x avg AND price < weekly EMA200
            elif price < dl and vol > 1.5 * vol_ma and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle OR price < weekly EMA200 (trend change)
            if price < dm or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle OR price > weekly EMA200 (trend change)
            if price > dm or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_WeeklyEMA200"
timeframe = "1d"
leverage = 1.0