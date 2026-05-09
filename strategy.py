#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1d EMA trend filter and volume confirmation
# Long when: Alligator jaws < teeth < lips (bullish alignment), 1d EMA(50) rising, volume spike
# Short when: Alligator jaws > teeth > lips (bearish alignment), 1d EMA(50) falling, volume spike
# Exit when: Alligator alignment breaks or trend reverses
# Williams Alligator uses smoothed medians (Jaw=13, Teeth=8, Lips=5) to filter noise.
# Works in trending markets (bull/bear) by catching sustained moves; avoids whipsaws in ranges.
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

name = "6h_WilliamsAlligator_1dEMA_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _sma(arr, window):
    """Simple moving average with NaN for insufficient data."""
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=float)
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).mean().values

def _smma(arr, window):
    """Smoothed moving average (SMMA) = EMA with alpha = 1/window."""
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=float)
    alpha = 1.0 / window
    s = pd.Series(arr)
    return s.ewm(alpha=alpha, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    median = (high + low) / 2
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA of median
    jaw = _smma(median, 13)
    teeth = _smma(median, 8)
    lips = _smma(median, 5)
    
    # Bullish: Lips > Teeth > Jaw; Bearish: Jaw > Teeth > Lips
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (jaw > teeth) & (teeth > lips)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alignment + 1d EMA rising + volume spike
            if (bullish_alignment[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + 1d EMA falling + volume spike
            elif (bearish_alignment[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment breaks or trend turns down
            if (not bullish_alignment[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks or trend turns up
            if (not bearish_alignment[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals