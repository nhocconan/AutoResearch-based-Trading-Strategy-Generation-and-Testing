#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Williams Fractal breakout + volume confirmation.
# Uses bearish/bullish fractals from 1D candles to identify potential reversal points.
# Breakouts are confirmed when price moves beyond the fractal level with above-average volume.
# Works in both bull and bear markets by capturing breakouts in either direction.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "12h_1d_WilliamsFractal_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractal calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals on 1d timeframe
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal: middle bar is highest
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True
            
        # Bullish fractal: middle bar is lowest
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Convert to price levels (use the fractal bar's high/low)
    bearish_level = np.where(bearish_fractal, high_1d, np.nan)
    bullish_level = np.where(bullish_fractal, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_level = pd.Series(bearish_level).ffill().values
    bullish_level = pd.Series(bullish_level).ffill().values
    
    # Align fractal levels to 12h timeframe (requires 2-bar confirmation delay for fractals)
    bearish_level_aligned = align_htf_to_ltf(prices, df_1d, bearish_level, additional_delay_bars=2)
    bullish_level_aligned = align_htf_to_ltf(prices, df_1d, bullish_level, additional_delay_bars=2)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_level_aligned[i]) or np.isnan(bullish_level_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above recent bearish fractal resistance with volume
            if close[i] > bearish_level_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below recent bullish fractal support with volume
            elif close[i] < bullish_level_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below recent bullish fractal support
            if close[i] < bullish_level_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above recent bearish fractal resistance
            if close[i] > bearish_level_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals