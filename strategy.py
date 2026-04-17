#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 12h EMA(50) is rising AND volume > 1.3x average.
Short when price breaks below Donchian lower band AND 12h EMA(50) is falling AND volume > 1.3x average.
Exit when price reverts to Donchian middle (20-period mean) or EMA(50) flips direction.
Uses 6h for Donchian calculation and 12h for EMA trend filter to reduce whipsaw.
Target: 50-150 total trades over 4 years (12-37/year). Donchian breakouts capture trends,
volume confirmation filters fakeouts, EMA trend filter avoids counter-trend trades.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian channels on 6h timeframe (20-period)
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    donchian_upper = high_6h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_6h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_50 = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # EMA slope (rising/falling) - positive = rising, negative = falling
    ema_slope = np.diff(ema_50, prepend=ema_50[0])
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_slope_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_middle_aligned[i]
        ema_slope_val = ema_slope_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND EMA(50) rising AND volume > 1.3x avg
            if price > du and ema_slope_val > 0 and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND EMA(50) falling AND volume > 1.3x avg
            elif price < dl and ema_slope_val < 0 and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle OR EMA(50) starts falling
            if price < dm or ema_slope_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle OR EMA(50) starts rising
            if price > dm or ema_slope_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_EMA50_Volume_Filter"
timeframe = "6h"
leverage = 1.0