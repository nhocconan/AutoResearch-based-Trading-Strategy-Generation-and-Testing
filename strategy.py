#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# Breakouts above/below 20-period Donchian channel with volume > 1.5x average and 12h HMA trend alignment
# Works in bull/bear markets: breakouts capture momentum, volume filter reduces false signals, HMA trend avoids counter-trend trades
# Designed for ~30-50 trades/year to minimize fee drag

name = "4h_12h_donchian_breakout_volume_hma_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21)
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hull_moving_average(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        
        wma_half = wma(values, half_window)
        wma_full = wma(values, window)
        
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full(len(values), np.nan)
        
        # Align arrays: wma_half starts at index half_window-1, wma_full at index window-1
        # We need to align them to the same index for subtraction
        diff = 2 * wma_half[-len(wma_full):] - wma_full
        hma = wma(diff, sqrt_window)
        
        # Pad with NaN at the beginning
        result = np.full(len(values), np.nan)
        start_idx = len(values) - len(hma)
        if start_idx >= 0:
            result[start_idx:] = hma
        return result
    
    hma_12h = hull_moving_average(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian(20) channels
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above upper Donchian with volume confirmation and 12h HMA uptrend
        long_breakout = close[i] > upper_20[i]
        long_volume = volume_confirmed[i]
        long_trend = close[i] > hma_12h_aligned[i]  # Price above 12h HMA = uptrend
        
        # Short conditions: price breaks below lower Donchian with volume confirmation and 12h HMA downtrend
        short_breakout = close[i] < lower_20[i]
        short_volume = volume_confirmed[i]
        short_trend = close[i] < hma_12h_aligned[i]  # Price below 12h HMA = downtrend
        
        if position == 1:  # Long position
            # Exit long if price closes below midpoint of Donchian channel or trend reverses
            midpoint = (upper_20[i] + lower_20[i]) / 2
            if close[i] < midpoint or not long_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price closes above midpoint of Donchian channel or trend reverses
            midpoint = (upper_20[i] + lower_20[i]) / 2
            if close[i] > midpoint or not short_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above upper Donchian with volume and trend confirmation
            if long_breakout and long_volume and long_trend:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below lower Donchian with volume and trend confirmation
            elif short_breakout and short_volume and short_trend:
                position = -1
                signals[i] = -0.25
    
    return signals