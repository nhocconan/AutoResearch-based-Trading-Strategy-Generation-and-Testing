#!/usr/bin/env python3
# 6h_elliott_wave_swing_sr_1d_volume_v1
# Hypothesis: Combine 1d swing-based support/resistance with 6s Elliott Wave structure.
# Long when price touches 1d swing low with bullish 6s impulse (price > 6s EMA21).
# Short when price touches 1d swing high with bearish 6s impulse (price < 6s EMA21).
# Volume confirmation > 1.5x average to avoid false breaks.
# Uses natural market structure (swings) which works in both bull/bear markets.
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability bounces.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elliott_wave_swing_sr_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for swing points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d swing highs and lows (fractal-based)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Swing high: high > previous 2 and next 2 highs
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = True
    
    # Get swing levels (only actual swing points)
    swing_high_levels = np.where(swing_high, high_1d, np.nan)
    swing_low_levels = np.where(swing_low, low_1d, np.nan)
    
    # Forward fill to get last swing level
    swing_high_levels = pd.Series(swing_high_levels).ffill().values
    swing_low_levels = pd.Series(swing_low_levels).ffill().values
    
    # Align swing levels to 6h timeframe
    swing_high_6h = align_htf_to_ltf(prices, df_1d, swing_high_levels)
    swing_low_6h = align_htf_to_ltf(prices, df_1d, swing_low_levels)
    
    # 6s EMA21 for impulse direction
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    bullish_impulse = close > ema21
    bearish_impulse = close < ema21
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(swing_high_6h[i]) or np.isnan(swing_low_6h[i]) or \
           np.isnan(ema21[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below swing low or opposite signal
            if close[i] < swing_low_6h[i] or \
               (close[i] < ema21[i] and volume[i] > 1.5 * avg_volume[i] and bearish_impulse[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above swing high or opposite signal
            if close[i] > swing_high_6h[i] or \
               (close[i] > ema21[i] and volume[i] > 1.5 * avg_volume[i] and bullish_impulse[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price touches or crosses above swing low with bullish impulse
            if close[i] >= swing_low_6h[i] and volume_ok and bullish_impulse[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or crosses below swing high with bearish impulse
            elif close[i] <= swing_high_6h[i] and volume_ok and bearish_impulse[i]:
                position = -1
                signals[i] = -0.25
    
    return signals