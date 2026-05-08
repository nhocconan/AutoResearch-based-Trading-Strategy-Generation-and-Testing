#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on weekly close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: bearish (high) and bullish (low)
    n1 = len(high_1d)
    bearish_fractal = np.zeros(n1, dtype=bool)
    bullish_fractal = np.zeros(n1, dtype=bool)
    
    for i in range(2, n1 - 2):
        # Bearish fractal: middle bar highest of 5
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: middle bar lowest of 5
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Convert to float arrays for alignment
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal_float, additional_delay_bars=2)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal_float, additional_delay_bars=2)
    
    # Volume confirmation: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_6h[i]) or np.isnan(bullish_fractal_6h[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish fractal + above weekly EMA50 + volume
            if (bullish_fractal_6h[i] > 0.5 and
                close[i] > ema_50_1w_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal + below weekly EMA50 + volume
            elif (bearish_fractal_6h[i] > 0.5 and
                  close[i] < ema_50_1w_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish fractal or below weekly EMA50
            if (bearish_fractal_6h[i] > 0.5 or
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish fractal or above weekly EMA50
            if (bullish_fractal_6h[i] > 0.5 or
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals