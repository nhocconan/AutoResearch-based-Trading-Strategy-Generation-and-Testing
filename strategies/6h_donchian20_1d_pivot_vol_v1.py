#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with daily pivot direction and volume confirmation.
# Uses 1-day pivot points to establish bias (long above pivot, short below pivot).
# Breakouts in direction of pivot bias with volume confirmation capture institutional moves.
# Designed for 6h timeframe to target 50-150 trades over 4 years with medium frequency.
# Works in bull/bear markets via pivot-based directional bias.

name = "6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day pivot points (calculated from prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculations (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 6h timeframe (shifted by 1 day for no look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Pivot bias: long above pivot, short below pivot
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below pivot or stoploss
            if (close[i] < pivot_aligned[i] or 
                close[i] < entry_price - 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above pivot or stoploss
            if (close[i] > pivot_aligned[i] or 
                close[i] > entry_price + 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of pivot bias
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > r1_aligned[i] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < s1_aligned[i] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals