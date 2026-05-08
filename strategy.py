#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w trend filter and volume confirmation.
# Uses weekly trend (1w EMA20) to filter direction and daily Williams Fractals for breakout signals.
# Long when bullish fractal breaks above weekly EMA20 AND volume > 2x 20-period average.
# Short when bearish fractal breaks below weekly EMA20 AND volume > 2x 20-period average.
# Exit when price crosses back below/above weekly EMA20.
# Williams Fractals identify potential turning points with built-in confirmation (requires 2 bars after).
# Weekly trend filter ensures we trade with higher timeframe momentum.
# Volume confirmation filters weak breakouts. Target: 50-150 total trades over 4 years.

name = "6h_WilliamsFractal_1wEMA20_Volume"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Fractals: need 5 bars (2 left, center, 2 right)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bullish_fractal = np.zeros(len(df_1d), dtype=bool)
    bearish_fractal = np.zeros(len(df_1d), dtype=bool)
    
    # Calculate fractals: bullish = low point with two higher lows on each side
    # bearish = high point with two lower highs on each side
    for i in range(2, len(df_1d) - 2):
        # Bullish fractal: lowest low in middle
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
        # Bearish fractal: highest high in middle
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish fractal signal AND price above weekly EMA20 AND volume filter
            long_cond = bullish_fractal_aligned[i] and (close[i] > ema20_1w_aligned[i]) and volume_filter[i]
            # Short conditions: bearish fractal signal AND price below weekly EMA20 AND volume filter
            short_cond = bearish_fractal_aligned[i] and (close[i] < ema20_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below weekly EMA20
            if close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above weekly EMA20
            if close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals