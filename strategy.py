#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 12h HMA > previous 12h HMA AND volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian lower band AND 12h HMA < previous 12h HMA AND volume > 1.5x 20-period average
# Exit when price crosses 4h Donchian middle band (mean reversion) OR 12h HMA flips direction
# Uses 4h primary timeframe with 12h HTF for HMA trend filter (more stable than 4h) and volume confirmation
# Higher timeframe HMA reduces whipsaw in ranging markets while capturing strong trends
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_12hHMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_12h, half_length)
    wma_full = wma(close_12h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_12h = wma(raw_hma, sqrt_length)
    
    # Pad to match original length
    hma_12h_padded = np.full(len(close_12h), np.nan)
    hma_12h_padded[half_length + sqrt_length - 1:] = hma_12h
    
    # Align HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    
    # Calculate 4h Donchian(20) channels
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
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
            # Long conditions: price breaks above Donchian upper AND 12h HMA rising AND volume spike
            if (close[i] > donchian_upper[i] and 
                hma_12h_aligned[i] > hma_12h_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 12h HMA falling AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  hma_12h_aligned[i] < hma_12h_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle (mean reversion) OR 12h HMA starts falling
            if close[i] < donchian_middle[i] or hma_12h_aligned[i] < hma_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle (mean reversion) OR 12h HMA starts rising
            if close[i] > donchian_middle[i] or hma_12h_aligned[i] > hma_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals