#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly EMA34 trend filter and volume spike confirmation
# Uses completed weekly EMA34 for trend direction (long above EMA34, short below)
# Enters on bullish/bearish Williams fractal breakouts with volume confirmation (>1.5x 20-period volume EMA)
# Williams fractals require 2-bar confirmation delay for completed pattern
# Designed for 12-37 trades/year on 6h timeframe to minimize fee drag
# Works in bull markets via upside fractal breaks and in bear markets via downside breaks with trend filter

name = "6h_WilliamsFractal_1wEMA34_VolumeSpike_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend filter and Williams fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA34 trend filter from prior completed weekly bar
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_shifted = np.roll(ema34_1w, 1)
    ema34_1w_shifted[0] = np.nan
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w_shifted)
    
    # Calculate weekly Williams fractals (requires 2-bar confirmation delay)
    # Bullish fractal: low[n-2] < low[n-1] and low[n] < low[n-1] and low[n+1] < low[n-1] and low[n+2] < low[n-1]
    # Bearish fractal: high[n-2] > high[n-1] and high[n] > high[n-1] and high[n+1] > high[n-1] and high[n+2] > high[n-1]
    n_1w = len(high_1w)
    bullish_fractal = np.full(n_1w, np.nan)
    bearish_fractal = np.full(n_1w, np.nan)
    
    for i in range(2, n_1w - 2):
        # Bullish fractal at i
        if (low_1w[i-2] < low_1w[i-1] and low_1w[i] < low_1w[i-1] and 
            low_1w[i+1] < low_1w[i-1] and low_1w[i+2] < low_1w[i-1]):
            bullish_fractal[i] = low_1w[i-1]  # fractal high point
        # Bearish fractal at i
        if (high_1w[i-2] > high_1w[i-1] and high_1w[i] > high_1w[i-1] and 
            high_1w[i+1] > high_1w[i-1] and high_1w[i+2] > high_1w[i-1]):
            bearish_fractal[i] = high_1w[i-1]  # fractal low point
    
    # Align fractals with 2-bar additional delay for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish fractal breakout AND price above weekly EMA34 AND volume spike
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish fractal breakdown AND price below weekly EMA34 AND volume spike
            elif (not np.isnan(bearish_fractal_aligned[i]) and 
                  close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly EMA34 OR below bearish fractal (if exists)
            exit_condition = close[i] < ema34_1w_aligned[i]
            if not np.isnan(bearish_fractal_aligned[i]):
                exit_condition = exit_condition or (close[i] < bearish_fractal_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly EMA34 OR above bullish fractal (if exists)
            exit_condition = close[i] > ema34_1w_aligned[i]
            if not np.isnan(bullish_fractal_aligned[i]):
                exit_condition = exit_condition or (close[i] > bullish_fractal_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals