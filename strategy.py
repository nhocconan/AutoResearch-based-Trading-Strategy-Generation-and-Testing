#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_fractal_breakout_v1
# William's fractals on weekly chart to identify key reversal points.
# Breakouts above weekly bearish fractal (resistance) or below weekly bullish fractal (support)
# with volume confirmation. Works in both bull and bear markets by trading breakouts
# from significant weekly levels. Target: 15-30 trades/year per symbol.
name = "6h_1w_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for fractal calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals (5-bar pattern)
    high_vals = df_1w['high'].values
    low_vals = df_1w['low'].values
    
    bearish_fractal = np.full(len(high_vals), np.nan)
    bullish_fractal = np.full(len(low_vals), np.nan)
    
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    for i in range(2, len(high_vals) - 2):
        if (high_vals[i-2] < high_vals[i-1] and 
            high_vals[i] < high_vals[i-1] and
            high_vals[i-3] < high_vals[i-1] and
            high_vals[i+1] < high_vals[i-1]):
            bearish_fractal[i-1] = high_vals[i-1]
    
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    for i in range(2, len(low_vals) - 2):
        if (low_vals[i-2] > low_vals[i-1] and 
            low_vals[i] > low_vals[i-1] and
            low_vals[i-3] > low_vals[i-1] and
            low_vals[i+1] > low_vals[i-1]):
            bullish_fractal[i-1] = low_vals[i-1]
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.3 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if fractal levels not ready
        if np.isnan(bearish_fractal_aligned[i]) and np.isnan(bullish_fractal_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly bearish fractal (resistance)
        if not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly bullish fractal (support)
        elif not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif not np.isnan(bullish_fractal_aligned[i]) and close[i] > bullish_fractal_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif not np.isnan(bearish_fractal_aligned[i]) and close[i] < bearish_fractal_aligned[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals