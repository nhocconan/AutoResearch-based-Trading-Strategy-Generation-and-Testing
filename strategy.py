#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 200-period SMA trend filter combined with 20-period Donchian breakout and volume confirmation.
# Long when price > SMA200 AND breaks above Donchian high AND volume > 1.5x average.
# Short when price < SMA200 AND breaks below Donchian low AND volume > 1.5x average.
# Exit when price crosses back inside Donchian channel.
# Designed to work in both bull and bear markets by using SMA200 as a long-term trend filter.
# Targets 20-50 trades per year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for SMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate SMA200 on daily timeframe
    close_1d = df_1d['close'].values
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (200 for SMA + buffer)
    start = 210
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(sma200_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price above SMA200 AND breakout above Donchian high AND volume confirmation
            if (price > sma200_1d_aligned[i] and price > high_20[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price below SMA200 AND breakdown below Donchian low AND volume confirmation
            elif (price < sma200_1d_aligned[i] and price < low_20[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Donchian low (opposite band)
            if price < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high (opposite band)
            if price > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_SMA200_Donchian_Volume"
timeframe = "4h"
leverage = 1.0