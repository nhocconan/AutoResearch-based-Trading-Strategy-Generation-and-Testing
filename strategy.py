#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with volume confirmation and daily pivot point filter
# Long when price breaks above 6h Donchian upper band with volume >1.5x 20-period average and price above daily pivot
# Short when price breaks below 6h Donchian lower band with volume >1.5x 20-period average and price below daily pivot
# Exit when price crosses the 6h Donchian midline
# Daily pivot point acts as a trend filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h and daily data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 6h Donchian channel (20-period lookback)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate daily pivot point (standard formula)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    pivot_point = (high_daily + low_daily + close_daily) / 3
    
    # Calculate 6h volume average (20-period)
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot_point)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_6h_current = volume[i]  # Current 6h volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and price above daily pivot
            if (price > donchian_upper_aligned[i] and 
                vol_6h_current > 1.5 * vol_ma_6h_aligned[i] and  # Volume confirmation
                price > pivot_aligned[i]):                      # Price above daily pivot for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and price below daily pivot
            elif (price < donchian_lower_aligned[i] and 
                  vol_6h_current > 1.5 * vol_ma_6h_aligned[i] and  # Volume confirmation
                  price < pivot_aligned[i]):                      # Price below daily pivot for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Donchian_DailyPivot_Volume"
timeframe = "6h"
leverage = 1.0