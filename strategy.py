#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper channel AND 12h HMA trending up AND volume > 2x 20-period average
# Short when price breaks below 4h Donchian lower channel AND 12h HMA trending down AND volume > 2x 20-period average
# Exit when price crosses 4h Donchian middle channel (mean reversion) OR 12h HMA flattens
# Uses 4h primary timeframe with 12h HTF for HMA trend filter
# Donchian channels provide clear breakout zones based on price action
# HMA filter ensures we only trade in trending markets, reducing whipsaw in ranges
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.30) to limit fee drag and manage drawdown
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_12hHMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # WMA function
    def wma(data, window):
        if len(data) < window:
            return np.full_like(data, np.nan)
        weights = np.arange(1, window + 1)
        result = np.convolve(data, weights, mode='valid') / weights.sum()
        padded = np.full_like(data, np.nan)
        padded[window-1:] = result
        return padded
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_12h, half_length)
    wma_full = wma(close_12h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_12h = wma(raw_hma, sqrt_length)
    
    # Align HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h Donchian(20) channels
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(hma_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND HMA trending up AND volume spike
            if (close[i] > donchian_upper[i] and 
                hma_12h_aligned[i] > hma_12h_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower AND HMA trending down AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  hma_12h_aligned[i] < hma_12h_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle (mean reversion) OR HMA flattens/down
            if close[i] < donchian_middle[i] or hma_12h_aligned[i] <= hma_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian middle (mean reversion) OR HMA flattens/up
            if close[i] > donchian_middle[i] or hma_12h_aligned[i] >= hma_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals