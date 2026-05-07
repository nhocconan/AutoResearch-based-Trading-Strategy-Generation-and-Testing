#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal Breakout with 1d EMA34 trend filter and volume confirmation.
# Uses daily bearish/bullish fractals to identify potential reversal points.
# Long when price breaks above bearish fractal resistance, price > 1d EMA34, and volume > 1.5x average.
# Short when price breaks below bullish fractal support, price < 1d EMA34, and volume > 1.5x average.
# Exit when price returns to the 1d EMA34 level or volume drops below average.
# Designed for 4h timeframe with target 20-50 trades/year to avoid fee drag.
# Williams Fractals require 2-bar confirmation, so we use additional_delay_bars=2.
name = "4h_WilliamsFractal_Breakout_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 1d timeframe (requires 2-bar confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n] and high[n+2] < high[n]
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+2] < high_1d[i]):
            bearish_fractal[i] = True
    
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n] and low[n+2] > low[n]
    for i in range(2, len(low_1d) - 2):
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+2] > low_1d[i]):
            bullish_fractal[i] = True
    
    # Get fractal price levels
    bearish_fractal_level = np.where(bearish_fractal, high_1d, np.nan)
    bullish_fractal_level = np.where(bullish_fractal, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_fractal_level = pd.Series(bearish_fractal_level).ffill().bfill().values
    bullish_fractal_level = pd.Series(bullish_fractal_level).ffill().bfill().values
    
    # Apply 2-bar additional delay for confirmation (Williams Fractals need 2 bars after)
    bearish_fractal_level_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_level, additional_delay_bars=2)
    bullish_fractal_level_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_level, additional_delay_bars=2)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_level_aligned[i]) or 
            np.isnan(bullish_fractal_level_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bearish fractal resistance, price > 1d EMA34, volume filter
            long_cond = (close[i] > bearish_fractal_level_aligned[i]) and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below bullish fractal support, price < 1d EMA34, volume filter
            short_cond = (close[i] < bullish_fractal_level_aligned[i]) and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to 1d EMA34 or volume filter fails
            if close[i] <= ema34_1d_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to 1d EMA34 or volume filter fails
            if close[i] >= ema34_1d_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals