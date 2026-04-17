#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND weekly EMA200 is rising AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND weekly EMA200 is falling AND volume > 1.5x average.
Exit when price reverts to Donchian middle (20-period mean).
Uses 1d for Donchian calculation and 1w for EMA200 trend filter to capture major trends while avoiding whipsaw.
Target: 30-100 total trades over 4 years (7-25/year). Donchian breakouts capture strong moves,
weekly EMA200 ensures we trade with the major trend, volume confirmation filters fakeouts.
Works in bull markets (captures uptrends with rising weekly EMA) and bear markets (captures downtrends with falling weekly EMA).
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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donchian_upper = high_1d_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1d_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w timeframe
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate EMA200 slope (rising/falling) on 1w timeframe
    ema200_slope = np.diff(ema200_1w_aligned, prepend=ema200_1w_aligned[0])
    ema200_rising = ema200_slope > 0
    ema200_falling = ema200_slope < 0
    
    # Volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper[i]
        dl = donchian_lower[i]
        dm = donchian_middle[i]
        ema200_val = ema200_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND weekly EMA200 rising AND volume > 1.5x avg
            if price > du and ema200_rising[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND weekly EMA200 falling AND volume > 1.5x avg
            elif price < dl and ema200_falling[i] and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle
            if price < dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle
            if price > dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA200_VolumeConfirm"
timeframe = "1d"
leverage = 1.0