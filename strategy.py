#!/usr/bin/env python3
# 6h_1d_williams_fractal_breakout_v1
# Hypothesis: Breakout trading based on Williams Fractals on daily timeframe with volume confirmation on 6h.
# Bullish fractal (higher low pattern) indicates potential support; breakout above recent high with volume triggers long.
# Bearish fractal (lower high pattern) indicates potential resistance; breakdown below recent low with volume triggers short.
# Works in both bull and bear markets by trading breakouts of fractal-identified support/resistance levels.
# Uses volume > 2x average to confirm breakout strength and avoid false signals.
# Targets 12-37 trades/year (50-150 total over 4 years) on 6h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True
        if (low_1d[i-2] > low_1d[i-1] and low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Convert to float arrays for alignment (1.0 where fractal exists, 0.0 otherwise)
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    # Williams fractals need 2 additional bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_float, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_float, additional_delay_bars=2)
    
    # Calculate recent high/low for breakout levels (using 10-period lookback)
    def calculate_recent_extremes(arr, lookback=10):
        recent_high = np.full_like(arr, np.nan)
        recent_low = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr)):
            recent_high[i] = np.max(arr[i-lookback:i])
            recent_low[i] = np.min(arr[i-lookback:i])
        return recent_high, recent_low
    
    recent_high_1d, recent_low_1d = calculate_recent_extremes(close_1d, 10)
    recent_high_aligned = align_htf_to_ltf(prices, df_1d, recent_high_1d)
    recent_low_aligned = align_htf_to_ltf(prices, df_1d, recent_low_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(recent_high_aligned[i]) or np.isnan(recent_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below recent low
            if close[i] < recent_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above recent high
            if close[i] > recent_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish fractal confirmed AND price breaks above recent high with volume surge
            if (bullish_fractal_aligned[i] > 0.5 and  # Bullish fractal present
                close[i] > recent_high_aligned[i] and vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish fractal confirmed AND price breaks below recent low with volume surge
            elif (bearish_fractal_aligned[i] > 0.5 and  # Bearish fractal present
                  close[i] < recent_low_aligned[i] and vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals