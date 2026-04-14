#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Confirmation and 1D Trend Filter
# Uses 4h Donchian channel breakouts as primary signal - proven to work in trending markets
# Volume confirmation ensures breakout strength, 1D EMA filter avoids counter-trend trades
# Designed for 4h timeframe with target of 25-35 trades/year (100-140 total over 4 years)
# Works in both bull and bear markets by following trend direction from higher timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian channel
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        
        # Skip if volume data is not available
        if np.isnan(vol_ma):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume and uptrend filter
            if price > highest_high[i] and volume_confirm and price > ema_1d_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower band with volume and downtrend filter
            elif price < lowest_low[i] and volume_confirm and price < ema_1d_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian lower band or trend changes
            if price < lowest_low[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian upper band or trend changes
            if price > highest_high[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_1DTrend"
timeframe = "4h"
leverage = 1.0