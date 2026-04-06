#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# Long when price breaks above 20-bar high + price > weekly pivot + volume > 1.5x average
# Short when price breaks below 20-bar low + price < weekly pivot + volume > 1.5x average
# Weekly pivot from daily data: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
# Uses weekly pivot for trend bias to avoid counter-trend trades
# Volume confirmation reduces false signals
# Target: 50-150 total trades over 4 years with controlled risk

name = "6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 3:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We'll approximate using daily data: use highest high, lowest low, last close of the week
    # For simplicity, we use daily OHLC to calculate daily pivot then use it as weekly bias
    # In practice, we'd resample to weekly, but using daily pivot as proxy for weekly bias
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point: P = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align 1d pivot to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Donchian channels (20-period) on 6h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or trend changes (below pivot)
            elif close[i] < low_min[i] or close[i] < pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or trend changes (above pivot)
            elif close[i] > high_max[i] or close[i] > pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price breaks above Donchian high + above pivot + volume spike
            if (close[i] > high_max[i] and close[i] > pivot_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + below pivot + volume spike
            elif (close[i] < low_min[i] and close[i] < pivot_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals