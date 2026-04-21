#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Williams Alligator for trend direction and 12h Williams Fractals for breakout entries.
The Williams Alligator identifies strong trends using smoothed moving averages (Jaws, Teeth, Lips).
Williams Fractals identify potential breakout points where price moves beyond recent highs/lows.
Combining these with volume confirmation creates a robust trend-following system that works in both bull and bear markets.
Target: 15-35 trades/year to minimize fee drag while capturing significant trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    smma = np.full_like(data, np.nan)
    smma[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def williams_alligator(high, low, close):
    """Calculate Williams Alligator lines"""
    median_price = (high + low) / 2
    jaws = calculate_smma(median_price, 13)  # Blue line
    teeth = calculate_smma(median_price, 8)   # Red line
    lips = calculate_smma(median_price, 5)    # Green line
    return jaws, teeth, lips

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals (bearish and bullish)"""
    n = len(high)
    bearish = np.zeros(n, dtype=bool)
    bullish = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # Bearish fractal: high is higher than 2 bars on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
        # Bullish fractal: low is lower than 2 bars on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Alligator trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    jaws_1d, teeth_1d, lips_1d = williams_alligator(high_1d, low_1d, close_1d)
    
    # Determine trend: bullish when Lips > Teeth > Jaws, bearish when Lips < Teeth < Jaws
    bullish_trend = (lips_1d > teeth_1d) & (teeth_1d > jaws_1d)
    bearish_trend = (lips_1d < teeth_1d) & (teeth_1d < jaws_1d)
    
    # Align 1d trend to 12h timeframe
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate 12h Williams Fractals for entry signals
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_12h, low_12h)
    
    # Volume confirmation: 12h volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in direction of 1d Alligator trend
        is_bullish_trend = bullish_trend_aligned[i] > 0.5
        is_bearish_trend = bearish_trend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + bullish fractal breakout + volume confirmation
            if (is_bullish_trend and bullish_fractal[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + bearish fractal breakout + volume confirmation
            elif (is_bearish_trend and bearish_fractal[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when trend changes or fractal in opposite direction appears
            if position == 1:
                # Exit long: bearish trend emerges or bearish fractal appears
                if is_bearish_trend or bearish_fractal[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish trend emerges or bullish fractal appears
                if is_bullish_trend or bullish_fractal[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Fractals_Volume"
timeframe = "12h"
leverage = 1.0