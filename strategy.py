#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation
# Long when price breaks above Donchian high + weekly pivot shows bullish bias + volume > 1.5x average
# Short when price breaks below Donchian low + weekly pivot shows bearish bias + volume > 1.5x average
# Exit when price crosses Donchian midpoint or weekly pivot flips bias
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Weekly pivot from 1d provides structural bias to avoid counter-trend trades in ranging markets

name = "6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 6h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Weekly pivot from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot (using prior week's OHLC)
    # We'll use prior daily values to calculate weekly pivot
    pivot = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)
    r2 = np.zeros_like(close_1d)
    s2 = np.zeros_like(close_1d)
    r3 = np.zeros_like(close_1d)
    s3 = np.zeros_like(close_1d)
    r4 = np.zeros_like(close_1d)
    s4 = np.zeros_like(close_1d)
    
    # Calculate pivot points for each 1d bar
    for i in range(1, len(close_1d)):
        # Use previous day's data for pivot calculation (no look-ahead)
        if i >= 1:
            pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
            pivot[i] = pp
            r1[i] = 2 * pp - low_1d[i-1]
            s1[i] = 2 * pp - high_1d[i-1]
            r2[i] = pp + (high_1d[i-1] - low_1d[i-1])
            s2[i] = pp - (high_1d[i-1] - low_1d[i-1])
            r3[i] = high_1d[i-1] + 2 * (pp - low_1d[i-1])
            s3[i] = low_1d[i-1] - 2 * (high_1d[i-1] - pp)
            r4[i] = pp + 3 * (high_1d[i-1] - low_1d[i-1])
            s4[i] = pp - 3 * (high_1d[i-1] - low_1d[i-1])
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Determine weekly pivot bias
    # Bullish bias: price above weekly pivot
    # Bearish bias: price below weekly pivot
    bullish_bias = close_1d > pivot
    bearish_bias = close_1d < pivot
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1d, bullish_bias.astype(float))
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1d, bearish_bias.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(pivot_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit if price crosses Donchian midpoint OR weekly pivot turns bearish
            if close[i] <= donch_mid[i] or bearish_bias_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit if price crosses Donchian midpoint OR weekly pivot turns bullish
            if close[i] >= donch_mid[i] or bullish_bias_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with pivot bias and volume confirmation
            # Bullish breakout: price above Donchian high + bullish pivot bias + volume
            if (close[i] > donch_high[i] and 
                bullish_bias_aligned[i] > 0.5 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + bearish pivot bias + volume
            elif (close[i] < donch_low[i] and 
                  bearish_bias_aligned[i] > 0.5 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals