#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian channel breakout with 1-day pivot point confirmation and volume filter.
# Long when price breaks above 6h Donchian high(20) AND daily pivot > daily open (bullish bias).
# Short when price breaks below 6h Donchian low(20) AND daily pivot < daily open (bearish bias).
# Volume confirmation: current volume > 1.5x 6h volume average.
# Uses discrete position sizing (0.25) to limit drawdown and reduce trade frequency.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years.

name = "6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):  # 20-period lookback
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 1-day pivot point data (daily high, low, close, open)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Daily pivot point: (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Daily bias: pivot > open = bullish, pivot < open = bearish
    daily_bullish = pivot_1d > open_1d
    daily_bearish = pivot_1d < open_1d
    
    # Align daily data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # 6-hour volume average (10-period)
    vol_ma = np.full(n, np.nan)
    for i in range(9, n):  # 10-period average
        vol_ma[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 9)  # Donchian needs 19, volume needs 9
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(daily_bullish_aligned[i]) or 
            np.isnan(daily_bearish_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 6h volume average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian low or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian high or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if volume_filter:
                # Long: break above Donchian high with bullish daily bias
                if (close[i] > highest_high[i] and daily_bullish_aligned[i] > 0.5):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below Donchian low with bearish daily bias
                elif (close[i] < lowest_low[i] and daily_bearish_aligned[i] > 0.5):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals