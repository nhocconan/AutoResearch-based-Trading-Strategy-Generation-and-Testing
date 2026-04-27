#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Williams Fractals identify potential reversal points: bearish fractal = potential resistance, bullish fractal = potential support.
# We trade breakouts of these fractal levels in the direction of the 1d EMA34 trend.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Fractals require 2-bar confirmation, so we use additional_delay_bars=2 when aligning.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: 5-bar pattern (high[2] is highest of 5)
    # Bearish fractal: high[2] > high[1] and high[2] > high[3] and high[2] > high[0] and high[2] > high[4]
    # Bullish fractal: low[2] < low[1] and low[2] < low[3] and low[2] < low[0] and low[2] < low[4]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    # Fractals need 2-bar confirmation delay (already built into fractal calculation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in direction of 1d trend
        if close[i] > ema34_1d_aligned[i] and volume_filter[i]:  # Uptrend
            # Buy breakout above bearish fractal (resistance)
            if not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Exit long when price breaks below bullish fractal (support)
            elif position == 1 and not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif close[i] < ema34_1d_aligned[i] and volume_filter[i]:  # Downtrend
            # Sell breakdown below bullish fractal (support)
            if not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i]:
                signals[i] = -0.25
                position = -1
            # Exit short when price breaks above bearish fractal (resistance)
            elif position == -1 and not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
        else:
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0