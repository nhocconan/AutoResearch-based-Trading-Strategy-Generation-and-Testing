#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h chart with 1w Williams Fractal filter and 1d pivot point breakout.
# Long when price breaks above R1 pivot with bullish weekly fractal confirmation.
# Short when price breaks below S1 pivot with bearish weekly fractal confirmation.
# Williams Fractals require 2-bar confirmation (center bar + 2 bars after) to avoid look-ahead.
# Uses weekly fractals to filter trend direction and avoid counter-trend trades.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's pivot points (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1w data for Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractals: need 5 points (n-2, n-1, n, n+1, n+2)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] < high[n+1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] > low[n+1] < low[n+2]
    n_w = len(high_1w)
    bearish_fractal = np.zeros(n_w, dtype=bool)
    bullish_fractal = np.zeros(n_w, dtype=bool)
    
    for i in range(2, n_w - 2):
        # Bearish fractal pattern
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i-1] > high_1w[i] and 
            high_1w[i] < high_1w[i+1] and 
            high_1w[i+1] > high_1w[i+2]):
            bearish_fractal[i] = True
        
        # Bullish fractal pattern
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i-1] < low_1w[i] and 
            low_1w[i] > low_1w[i+1] and 
            low_1w[i+1] < low_1w[i+2]):
            bullish_fractal[i] = True
    
    # Convert to float arrays (1.0 where fractal exists, 0.0 otherwise)
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    
    # Align fractals to 6h timeframe with 2-bar confirmation delay
    # Williams fractals need 2 additional bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal_float, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal_float, additional_delay_bars=2)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i] > 0.5  # True if fractal present
        bullish_fractal_val = bullish_fractal_aligned[i] > 0.5   # True if fractal present
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above R1, bullish weekly fractal, volume
            if price > r1_val and bullish_fractal_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, bearish weekly fractal, volume
            elif price < s1_val and bearish_fractal_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or bearish fractal appears
            if price < s1_val or bearish_fractal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or bullish fractal appears
            if price > r1_val or bullish_fractal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_WilliamsFractal_1d_Pivot_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0