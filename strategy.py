#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above Donchian upper band AND volume > 1.3x average AND 12h EMA34 > EMA50 (bullish).
Short when price breaks below Donchian lower band AND volume > 1.3x average AND 12h EMA34 < EMA50 (bearish).
Exit when price reverts to Donchian middle (20-period mean).
Uses 4h for Donchian calculation and volume, 12h for EMA trend filter to reduce whipsaw.
Target: 75-200 total trades over 4 years (19-50/year). Donchian breakouts capture trends,
volume confirmation filters fakeouts, EMA filter avoids counter-trend trades.
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
    
    # Get 4h data for Donchian calculation and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema34 = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema50 = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h Donchian to 4h timeframe (no alignment needed)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    donchian_middle_aligned = donchian_middle
    
    # Align 12h EMAs to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_middle_aligned[i]
        ema34_val = ema34_aligned[i]
        ema50_val = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 1.3x avg AND 12h EMA34 > EMA50 (bullish)
            if price > du and vol > 1.3 * vol_ma and ema34_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 1.3x avg AND 12h EMA34 < EMA50 (bearish)
            elif price < dl and vol > 1.3 * vol_ma and ema34_val < ema50_val:
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

name = "4h_Donchian20_Volume_12hEMA_Filter"
timeframe = "4h"
leverage = 1.0