#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band AND 1w HMA(21) is rising AND volume > 1.5x 20-period average
# Short when price breaks below 1d Donchian lower band AND 1w HMA(21) is falling AND volume > 1.5x 20-period average
# Exit when price crosses 1d Donchian middle band (mean reversion)
# Uses 1d primary timeframe with 1w HTF for HMA trend filter
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Donchian20_Breakout_1wHMA_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w HMA(21)
    close_1w = df_1w['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    if len(close_1w) >= 21:
        wma_half = np.array([wma(close_1w[i:i+half_length], half_length) 
                            for i in range(len(close_1w) - half_length + 1)])
        wma_full = np.array([wma(close_1w[i:i+21], 21) 
                            for i in range(len(close_1w) - 21 + 1)])
        wma_sqrt = np.array([wma(close_1w[i:i+sqrt_length], sqrt_length) 
                            for i in range(len(close_1w) - sqrt_length + 1)])
        
        # Align arrays to same length
        min_len = min(len(wma_half), len(wma_full), len(wma_sqrt))
        if min_len > 0:
            wma_half = wma_half[-min_len:]
            wma_full = wma_full[-min_len:]
            wma_sqrt = wma_sqrt[-min_len:]
            hma_1w = 2 * wma_half - wma_full
            hma_1w = wma(hma_1w, sqrt_length)
            
            # Pad to match original length
            hma_1w_padded = np.full(len(close_1w), np.nan)
            start_idx = len(close_1w) - len(hma_1w)
            hma_1w_padded[start_idx:] = hma_1w
            hma_1w = hma_1w_padded
        else:
            hma_1w = np.full(len(close_1w), np.nan)
    else:
        hma_1w = np.full(len(close_1w), np.nan)
    
    # Align HMA to 1d timeframe
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate HMA slope for trend filter (rising/falling)
    hma_slope = np.diff(hma_1w_aligned, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Calculate 1d Donchian channels (20-period)
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
        if (np.isnan(hma_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND HMA rising AND volume spike
            if (close[i] > donchian_upper[i] and 
                hma_rising[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND HMA falling AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  hma_falling[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals