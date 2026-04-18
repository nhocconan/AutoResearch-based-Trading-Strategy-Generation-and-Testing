#!/usr/bin/env python3
"""
4h Williams Fractal Breakout with 1d Trend Filter
Hypothesis: Williams fractals identify key support/resistance levels where price breaks out with momentum. Combined with 1d trend filter (EMA50) to avoid counter-trend trades and volume confirmation to ensure institutional participation, this captures strong breakout moves in both bull and bear markets. Works because fractals are lagging indicators that only form after price has already shown rejection at a level, making breakouts more meaningful.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    alpha = 2 / (period + 1)
    ema = np.zeros_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (up) and bullish (down)"""
    n = len(high)
    bearish = np.zeros(n, dtype=bool)  # peak fractal
    bullish = np.zeros(n, dtype=bool)   # trough fractal
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest among 5 bars
        if (high[i] > high[i-2] and high[i] > high[i-1] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
        # Bullish fractal: low[i] is lowest among 5 bars
        if (low[i] < low[i-2] and low[i] < low[i-1] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 4h data
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high, low)
    
    # Williams fractals need 2 extra bars for confirmation (they form after the fact)
    bearish_fractal_aligned = align_htf_to_ltf(prices, 
                                               pd.DataFrame({'high': high, 'low': low}), 
                                               bearish_fractal.astype(float),
                                               additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, 
                                               pd.DataFrame({'high': high, 'low': low}), 
                                               bullish_fractal.astype(float),
                                               additional_delay_bars=2)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1 = uptrend (price > EMA), -1 = downtrend (price < EMA)
        trend = 1 if close[i] > ema_1d_aligned[i] else -1
        
        if position == 0:
            # Enter long: bullish fractal breakout + volume + uptrend
            if (bullish_fractal_aligned[i] and 
                vol_confirm[i] and 
                trend == 1):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal breakout + volume + downtrend
            elif (bearish_fractal_aligned[i] and 
                  vol_confirm[i] and 
                  trend == -1):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below the bullish fractal level or trend changes
            # We use a simple exit: trend reversal
            if trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above the bearish fractal level or trend changes
            if trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dTrendFilter"
timeframe = "4h"
leverage = 1.0