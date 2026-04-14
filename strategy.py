#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price channel breakout (Donchian 10) with weekly ADX trend filter and volume confirmation
# Long when price breaks above Donchian(10) high AND weekly ADX > 25 AND volume > 1.5x 10-period average
# Short when price breaks below Donchian(10) low AND weekly ADX > 25 AND volume > 1.5x 10-period average
# Exit when price crosses back inside the Donchian channel (opposite band)
# Weekly ADX ensures we only trade in strong trends (avoiding chop), reducing false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels on 12h (10-period high/low)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate weekly ADX for trend filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])) > 
                       (np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w), 
                       np.maximum(high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w) > 
                        (high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])), 
                        np.maximum(np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w, 0), 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dx = 100 * np.abs(dm_plus - dm_minus) / (dm_plus + dm_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate volume average for confirmation (10-period)
    vol_avg = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (10 for Donchian + buffer)
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_10[i]) or np.isnan(low_10[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above Donchian high + strong trend (ADX>25) + volume confirmation
            if (price > high_10[i] and adx_1w_aligned[i] > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below Donchian low + strong trend (ADX>25) + volume confirmation
            elif (price < low_10[i] and adx_1w_aligned[i] > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Donchian low (opposite band)
            if price < low_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high (opposite band)
            if price > high_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_1wADX_Volume"
timeframe = "12h"
leverage = 1.0