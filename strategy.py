#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# Go long when price breaks above Donchian(20) high and price is above 1d weekly pivot (bullish bias)
# Go short when price breaks below Donchian(20) low and price is below 1d weekly pivot (bearish bias)
# Volume confirmation (1.5x average) to avoid false breakouts
# Weekly pivot calculated from prior week's OHLC (using 1d data)
# Target: 75-200 total trades over 4 years with controlled risk (max 0.30 position size)

name = "6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot from 1d OHLC (prior week's data)
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We approximate using rolling window on 1d data (5 trading days per week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-day rolling window on 1d data
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian20 warmup
        # Skip if required data not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or bearish pivot bias
            elif close[i] < lowest_20[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or bullish pivot bias
            elif close[i] > highest_20[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for breakout entries with volume confirmation
            # Long: break above Donchian high + bullish pivot bias + volume spike
            if (close[i] > highest_20[i] and 
                close[i] > weekly_pivot_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low + bearish pivot bias + volume spike
            elif (close[i] < lowest_20[i] and 
                  close[i] < weekly_pivot_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals