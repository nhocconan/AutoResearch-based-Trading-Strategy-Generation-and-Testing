#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend
Hypothesis: Williams fractals identify swing points on 1d chart; breakouts above bearish fractals or below bullish fractals with 1d trend filter capture momentum in both bull and bear markets. Williams fractals require 2-bar confirmation, so we use additional_delay_bars=2. Low-frequency signals (target 12-37/year) minimize fee drag. Uses 6h timeframe for execution with 1d trend and fractal confirmation.
"""

name = "6h_WilliamsFractal_Breakout_1dTrend"
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
    
    # Get daily data for trend and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    # Williams fractals require 2-bar confirmation after the center bar
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Ensure EMA is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above bearish fractal (resistance) AND uptrend (price > EMA50)
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below bullish fractal (support) AND downtrend (price < EMA50)
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below bullish fractal (support) OR trend reversal (price < EMA50)
                if close[i] < bullish_fractal_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above bearish fractal (resistance) OR trend reversal (price > EMA50)
                if close[i] > bearish_fractal_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals