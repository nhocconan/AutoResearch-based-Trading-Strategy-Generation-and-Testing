#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Williams Fractal breakout and volume confirmation.
# Williams Fractal identifies potential reversal points. A bullish fractal is a low with two higher lows on each side.
# A bearish fractal is a high with two lower highs on each side.
# Long: Price breaks above the highest bearish fractal high of the last 20 days + volume > 1.3x average volume (20-period).
# Short: Price breaks below the lowest bullish fractal low of the last 20 days + volume > 1.3x average volume.
# Uses 1d Williams Fractals for swing structure, 4h for execution with volume confirmation.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    bearish_fractal = np.full(len(high_1d), np.nan)  # stores fractal high values
    bullish_fractal = np.full(len(low_1d), np.nan)   # stores fractal low values
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high with two lower highs on each side
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: low with two higher lows on each side
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Calculate rolling highest bearish fractal and lowest bullish fractal (20-period)
    highest_bearish = np.full(len(high_1d), np.nan)
    lowest_bullish = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        window_bearish = bearish_fractal[i-20:i]
        window_bullish = bullish_fractal[i-20:i]
        valid_bearish = window_bearish[~np.isnan(window_bearish)]
        valid_bullish = window_bullish[~np.isnan(window_bullish)]
        if len(valid_bearish) > 0:
            highest_bearish[i] = np.max(valid_bearish)
        if len(valid_bullish) > 0:
            lowest_bullish[i] = np.min(valid_bullish)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d fractal levels to 4h (need 2-bar confirmation delay for fractals)
    highest_bearish_aligned = align_htf_to_ltf(prices, df_1d, highest_bearish, additional_delay_bars=2)
    lowest_bullish_aligned = align_htf_to_ltf(prices, df_1d, lowest_bullish, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_bearish_aligned[i]) or np.isnan(lowest_bullish_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        highest_bear = highest_bearish_aligned[i]
        lowest_bull = lowest_bullish_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price breaks above highest bearish fractal + volume confirmation
            if (price > highest_bear and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lowest bullish fractal + volume confirmation
            elif (price < lowest_bull and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lowest bullish fractal
            if price < lowest_bull:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above highest bearish fractal
            if price > highest_bear:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Williams_Fractal_Breakout_Volume"
timeframe = "4h"
leverage = 1.0