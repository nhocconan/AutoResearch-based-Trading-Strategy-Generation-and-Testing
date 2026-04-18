#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + Volume Spike + 1d EMA Trend Filter
Long: Bullish fractal breakout + volume spike + 1d EMA rising
Short: Bearish fractal breakdown + volume spike + 1d EMA falling
Williams Fractals identify key support/resistance levels, breakouts capture momentum,
volume confirms strength, and 1d EMA ensures alignment with higher timeframe trend.
Designed for 4h timeframe to capture breakouts in both bull and bear markets.
Target: 80-150 total trades over 4 years (20-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (up) and bullish (down)"""
    n = len(high)
    bearish = np.full(n, np.nan)  # bearish fractal (peak)
    bullish = np.full(n, np.nan)  # bullish fractal (trough)
    
    for i in range(2, n - 2):
        # Bearish fractal: high is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        # Bullish fractal: low is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 4h
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high, low)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d EMA slope for trend filter
    ema_slope = np.diff(ema_34_1d_aligned, prepend=ema_34_1d_aligned[0])
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 5  # need fractal calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_slope[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Bullish fractal breakout + volume spike + 1d EMA rising
            if (not np.isnan(bullish_fractal[i]) and 
                price > bullish_fractal[i] and 
                volume_spike[i] and 
                ema_slope[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal breakdown + volume spike + 1d EMA falling
            elif (not np.isnan(bearish_fractal[i]) and 
                  price < bearish_fractal[i] and 
                  volume_spike[i] and 
                  ema_slope[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish fractal breakdown OR price closes below bullish fractal
            if (not np.isnan(bearish_fractal[i]) and price < bearish_fractal[i]) or \
               (not np.isnan(bullish_fractal[i]) and price < bullish_fractal[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish fractal breakout OR price closes above bearish fractal
            if (not np.isnan(bullish_fractal[i]) and price > bullish_fractal[i]) or \
               (not np.isnan(bearish_fractal[i]) and price > bearish_fractal[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_Volume_EMA34"
timeframe = "4h"
leverage = 1.0