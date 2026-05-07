#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 1d trend filter and volume confirmation.
# Bullish fractal: lowest low with two higher lows on each side.
# Bearish fractal: highest high with two lower highs on each side.
# Long when price breaks above bearish fractal (resistance) AND 1d EMA50 up trending AND volume > 1.5x 20-period average.
# Short when price breaks below bullish fractal (support) AND 1d EMA50 down trending AND volume > 1.5x 20-period average.
# Exit when price returns inside the fractal level or volume drops below average.
# Designed for 6h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "6h_WilliamsFractal_Breakout_1dEMA50_VolumeFilter"
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
    
    # Williams Fractals: need 5-point window (2 left, center, 2 right)
    # Bearish fractal: highest high with two lower highs on each side
    # Bullish fractal: lowest low with two higher lows on each side
    n_fractal = 5
    half = n_fractal // 2  # 2
    
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    # Start from index 2 to n-3 to have 2 left and 2 right
    for i in range(half, n - half):
        # Check for bearish fractal: current high is highest in window
        window_high = high[i - half:i + half + 1]
        if high[i] == np.max(window_high):
            # Ensure it's strictly higher than neighbors (not equal)
            if high[i] > high[i - 2] and high[i] > high[i - 1] and \
               high[i] > high[i + 1] and high[i] > high[i + 2]:
                bearish_fractal[i] = high[i]
        
        # Check for bullish fractal: current low is lowest in window
        window_low = low[i - half:i + half + 1]
        if low[i] == np.min(window_low):
            # Ensure it's strictly lower than neighbors
            if low[i] < low[i - 2] and low[i] < low[i - 1] and \
               low[i] < low[i + 1] and low[i] < low[i + 2]:
                bullish_fractal[i] = low[i]
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # EMA50 direction (trend)
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal[i]) or np.isnan(bullish_fractal[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bearish fractal (resistance), EMA50 rising, volume filter
            long_cond = (close[i] > bearish_fractal[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below bullish fractal (support), EMA50 falling, volume filter
            short_cond = (close[i] < bullish_fractal[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below bearish fractal OR EMA50 falls OR volume filter fails
            if (close[i] < bearish_fractal[i]) or (not ema50_rising[i]) or (not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above bullish fractal OR EMA50 rises OR volume filter fails
            if (close[i] > bullish_fractal[i]) or (not ema50_falling[i]) or (not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals