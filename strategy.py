#!/usr/bin/env python3
"""
6h_rolling_fractal_mean_reversion_v1
Hypothesis: In 6h timeframe, price often reverts to the mean after fractal
extremes (local highs/lows). We use a rolling fractal high/low (5-bar window)
combined with 60-period SMA trend filter and volume confirmation. Works in
both bull and bear markets by fading extremes against the trend direction.
"""

import numpy as np
import pandas as pd

name = "6h_rolling_fractal_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Rolling fractal high/low (5-bar window: 2 bars each side)
    # Fractal high: highest high in window, fractal low: lowest low in window
    window = 5
    half = window // 2  # 2
    
    fractal_high = np.full(n, np.nan)
    fractal_low = np.full(n, np.nan)
    
    for i in range(half, n - half):
        window_high = np.max(high[i - half:i + half + 1])
        window_low = np.min(low[i - half:i + half + 1])
        fractal_high[i] = window_high
        fractal_low[i] = window_low
    
    # 60-period SMA for trend filter
    sma60 = pd.Series(close).rolling(window=60, min_periods=60).mean().values
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(fractal_high[i]) or np.isnan(fractal_low[i]) or 
            np.isnan(sma60[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches fractal high (mean reversion complete) OR
            # price breaks below fractal low with volume (breakdown)
            if close[i] >= fractal_high[i] or (close[i] <= fractal_low[i] and vol_confirm):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches fractal low (mean reversion complete) OR
            # price breaks above fractal high with volume (breakout)
            if close[i] <= fractal_low[i] or (close[i] >= fractal_high[i] and vol_confirm):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long: price at/fractal low in uptrend (price > SMA60)
            if (close[i] <= fractal_low[i] and 
                vol_confirm and 
                close[i] > sma60[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short: price at/fractal high in downtrend (price < SMA60)
            elif (close[i] >= fractal_high[i] and 
                  vol_confirm and 
                  close[i] < sma60[i]):
                position = -1
                signals[i] = -0.25
    
    return signals