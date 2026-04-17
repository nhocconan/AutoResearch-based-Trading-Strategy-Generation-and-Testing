#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
Long when price breaks above Donchian upper band AND volume > 1.5x 20-period average AND price > 1d EMA200.
Short when price breaks below Donchian lower band AND volume > 1.5x 20-period average AND price < 1d EMA200.
Exit when price reverts to Donchian midpoint.
Uses 6h for price/volume/Donchian, 1d for EMA200 trend filter to avoid whipsaw in ranging markets.
Targets 75-200 total trades over 4 years (19-50/year). Donchian channels provide clear breakout levels,
volume confirmation reduces fakeouts, 1d EMA200 ensures we trade with the higher timeframe trend.
Works in bull markets (captures uptrends with bullish 1d EMA200) and bear markets (captures downtrends with bearish 1d EMA200).
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
    
    # Get 6h data for Donchian channels and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian channels on 6h timeframe (20-period)
    high_series = pd.Series(high_6h)
    low_series = pd.Series(low_6h)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate volume average (20-period) on 6h
    volume_series = pd.Series(volume_6h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily timeframe
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 6h Donchian channels, volume MA, and daily EMA200 to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 1.5x avg AND price > daily EMA200 (bullish trend)
            if price > upper and vol > 1.5 * vol_ma and price > ema_200:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 1.5x avg AND price < daily EMA200 (bearish trend)
            elif price < lower and vol > 1.5 * vol_ma and price < ema_200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Volume_1dEMA200_Filter"
timeframe = "6h"
leverage = 1.0