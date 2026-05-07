#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend and volume spike confirmation.
# Long when bullish fractal forms AND price breaks above recent high AND 1d EMA34 rising AND volume > 2x average.
# Short when bearish fractal forms AND price breaks below recent low AND 1d EMA34 falling AND volume > 2x average.
# Exit when price crosses the opposite fractal level (bearish for long exit, bullish for short exit).
# Williams Fractals identify potential turning points; combining with trend filter and volume spike
# captures momentum after consolidation. Works in both bull/bear by following 1d trend direction.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals (5-bar window: need 2 bars on each side)
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    highest = pd.Series(high).rolling(window=5, center=True).max().values
    lowest = pd.Series(low).rolling(window=5, center=True).min().values
    bearish_fractal = (high == highest)  # True at fractal point
    bullish_fractal = (low == lowest)    # True at fractal point
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish fractal AND price breaks above recent high AND 1d EMA34 rising AND volume spike
            long_cond = bullish_fractal[i] and (close[i] > highest[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: bearish fractal AND price breaks below recent low AND 1d EMA34 falling AND volume spike
            short_cond = bearish_fractal[i] and (close[i] < lowest[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below recent low (bearish fractal level)
            if close[i] < lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above recent high (bullish fractal level)
            if close[i] > highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals