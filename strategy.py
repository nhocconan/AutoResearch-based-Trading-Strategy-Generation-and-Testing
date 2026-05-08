#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h trend filter and volume confirmation.
# Long when price breaks above recent bearish fractal (resistance) AND 12h EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below recent bullish fractal (support) AND 12h EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the most recent fractal pair (between bullish and bearish fractal).
# Williams Fractals identify key swing points where price reverses. EMA50 filters higher timeframe trend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsFractal_Breakout_12hEMA50_Volume"
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
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Fractals: bearish (resistance) and bullish (support)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+3]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+3]
    high_vals = df_1d['high'].values
    low_vals = df_1d['low'].values
    n_1d = len(high_vals)
    
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_vals[i-2] < high_vals[i-1] and 
            high_vals[i] < high_vals[i-1] and
            high_vals[i-3] < high_vals[i-1] and
            high_vals[i+1] < high_vals[i-1] and
            high_vals[i+2] < high_vals[i-1]):
            bearish_fractal[i-1] = high_vals[i-1]
        
        if (low_vals[i-2] > low_vals[i-1] and 
            low_vals[i] > low_vals[i-1] and
            low_vals[i-3] > low_vals[i-1] and
            low_vals[i+1] > low_vals[i-1] and
            low_vals[i+2] > low_vals[i-1]):
            bullish_fractal[i-1] = low_vals[i-1]
    
    # Forward fill to get most recent fractal levels
    bearish_fractal = pd.Series(bearish_fractal).ffill().bfill().values
    bullish_fractal = pd.Series(bullish_fractal).ffill().bfill().values
    
    # Align fractal levels to 6h timeframe (need 2-bar confirmation delay for fractals)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 direction
    ema50_rising = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    ema50_falling[1:] = ema50_12h_aligned[1:] < ema50_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(55, 2)  # Sufficient warmup for EMA50 and fractals
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bearish fractal (resistance), 12h EMA50 rising, volume filter
            long_cond = (close[i] > bearish_fractal_aligned[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below bullish fractal (support), 12h EMA50 falling, volume filter
            short_cond = (close[i] < bullish_fractal_aligned[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below bullish fractal (support)
            if close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above bearish fractal (resistance)
            if close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals