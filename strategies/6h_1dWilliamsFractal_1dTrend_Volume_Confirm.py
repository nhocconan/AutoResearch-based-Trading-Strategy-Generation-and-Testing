#!/usr/bin/env python3
# 6h_1dWilliamsFractal_1dTrend_Volume_Confirm
# Uses daily Williams Fractals for support/resistance with daily trend filter and volume confirmation.
# Long when price breaks above bearish fractal resistance in uptrend, short when breaks below bullish fractal support in downtrend.
# Williams Fractals require 2-bar confirmation, so we use additional_delay_bars=2 in align_htf_to_ltf.
# Designed for 6h timeframe to capture institutional levels with trend alignment, working in both bull and bear markets.

name = "6h_1dWilliamsFractal_1dTrend_Volume_Confirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align Williams Fractals to 6h timeframe with 2-bar confirmation delay
    bearish_fractal_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(bearish_fractal_6h[i]) or np.isnan(bullish_fractal_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above bearish fractal resistance with uptrend and volume
            if close[i] > bearish_fractal_6h[i] and close[i] > ema_34_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below bullish fractal support with downtrend and volume
            elif close[i] < bullish_fractal_6h[i] and close[i] < ema_34_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below EMA34 or breaks below bullish fractal support
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry >= 2 and (close[i] < ema_34_6h[i] or close[i] < bullish_fractal_6h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA34 or breaks above bearish fractal resistance
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry >= 2 and (close[i] > ema_34_6h[i] or close[i] > bearish_fractal_6h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals