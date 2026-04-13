#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Williams Fractal reversal signals and volume confirmation.
# Bullish fractal (buy signal): price forms a bullish fractal (lowest low with two higher lows on each side) + volume > 1.5x average.
# Bearish fractal (sell signal): price forms a bearish fractal (highest high with two lower highs on each side) + volume > 1.5x average.
# Williams Fractals identify potential reversal points. Combined with volume confirmation and 6h execution, this should work in both bull and bear markets by catching reversals at key support/resistance levels.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

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
    # Bearish fractal: highest high with two lower highs on each side
    bearish_fractal = np.full(len(high_1d), np.nan)
    # Bullish fractal: lowest low with two higher lows on each side
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: current high is highest among 5 bars (2 left, 2 right)
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: current low is lowest among 5 bars (2 left, 2 right)
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Williams Fractals need 2 extra bars for confirmation (Williams fractals require 2 bars after the close to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: bullish fractal + volume confirmation
            if not np.isnan(bullish_fractal_val) and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: bearish fractal + volume confirmation
            elif not np.isnan(bearish_fractal_val) and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish fractal appears (potential top)
            if not np.isnan(bearish_fractal_val) and volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish fractal appears (potential bottom)
            if not np.isnan(bullish_fractal_val) and volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Williams_Fractal_Volume"
timeframe = "6h"
leverage = 1.0