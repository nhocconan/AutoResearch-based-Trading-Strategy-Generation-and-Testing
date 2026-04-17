#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume spike and 1d Williams %R oversold/overbought filter.
Long when price breaks above Donchian upper band AND volume > 1.5x average AND daily Williams %R < -80 (oversold).
Short when price breaks below Donchian lower band AND volume > 1.5x average AND daily Williams %R > -20 (overbought).
Exit when price reverts to Donchian middle OR daily Williams %R crosses back through -50 (mean reversion).
Uses 4h for price/volume, 1d for Williams %R filter to avoid buying strength/selling weakness.
Target: 75-200 total trades over 4 years (19-50/year). Donchian channels provide clear breakout levels,
volume confirmation reduces fakeouts, Williams %R ensures we buy into weakness and sell into strength.
Works in bull markets (captures uptrends from oversold) and bear markets (captures downtrends from overbought).
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
    
    # Get 1d data for Williams %R filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d timeframe (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), np.inf)
    
    # Align 4h Donchian channels, volume MA, and 1d Williams %R to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        williams_r_val = williams_r_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 1.5x avg AND daily Williams %R < -80 (oversold)
            if price > upper and vol > 1.5 * vol_ma and williams_r_val < -80:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 1.5x avg AND daily Williams %R > -20 (overbought)
            elif price < lower and vol > 1.5 * vol_ma and williams_r_val > -20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle OR daily Williams %R > -50 (no longer oversold)
            if price < middle or williams_r_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle OR daily Williams %R < -50 (no longer overbought)
            if price > middle or williams_r_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dWilliamsR_Filter"
timeframe = "4h"
leverage = 1.0