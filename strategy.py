#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume spike and 12h EMA trend filter.
Long when price breaks above Donchian upper band AND volume > 2.0x 20-period average AND price > 12h EMA34.
Short when price breaks below Donchian lower band AND volume > 2.0x 20-period average AND price < 12h EMA34.
Exit when price reverts to Donchian midpoint.
Uses 4h for price/volume/Donchian, 12h for EMA34 trend filter to avoid whipsaw in ranging markets.
Targets 75-150 total trades over 4 years (19-38/year). Donchian channels provide clear breakout levels,
volume confirmation reduces fakeouts, 12h EMA ensures we trade with the higher timeframe trend.
Works in bull markets (captures uptrends with bullish 12h EMA) and bear markets (captures downtrends with bearish 12h EMA).
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
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate volume average (20-period) on 4h
    volume_series = pd.Series(volume_4h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h Donchian channels, volume MA, and 12h EMA34 to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_34 = ema_34_12h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 2.0x avg AND price > 12h EMA34 (bullish trend)
            if price > upper and vol > 2.0 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 2.0x avg AND price < 12h EMA34 (bearish trend)
            elif price < lower and vol > 2.0 * vol_ma and price < ema_34:
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

name = "4h_Donchian20_Volume_12hEMA34_Filter"
timeframe = "4h"
leverage = 1.0