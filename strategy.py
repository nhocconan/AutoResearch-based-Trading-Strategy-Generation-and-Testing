#!/usr/bin/env python3
"""
6h_1d_1w_WilliamsFractal_TrendFilter_v1
Hypothesis: Trade breakouts from 1d Williams Fractals in the direction of 1w EMA200 trend.
In bull markets, buy breakouts above recent bearish fractal highs; in bear markets, sell breakdowns below recent bullish fractal lows.
Uses 6h for entry timing, 1d for fractal structure, 1w for trend filter.
Works in both bull/bear: trend filter ensures we only trade with the higher timeframe momentum.
Target: 12-25 trades/year per symbol (50-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 bars for fractals
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Compute Williams Fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    
    # Align fractals to 6h with 2-bar extra delay (needed for fractal confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Load 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Determine 1w trend: price above/below EMA200
        weekly_uptrend = price > ema_200_1w_aligned[i]
        weekly_downtrend = price < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above confirmed bearish fractal (resistance) in uptrend
            if (weekly_uptrend and 
                not np.isnan(bearish_fractal_aligned[i]) and 
                price > bearish_fractal_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below confirmed bullish fractal (support) in downtrend
            elif (weekly_downtrend and 
                  not np.isnan(bullish_fractal_aligned[i]) and 
                  price < bullish_fractal_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below bullish fractal (support) or weekly trend turns down
            if (not np.isnan(bullish_fractal_aligned[i]) and price < bullish_fractal_aligned[i]) or \
               (not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above bearish fractal (resistance) or weekly trend turns up
            if (not np.isnan(bearish_fractal_aligned[i]) and price > bearish_fractal_aligned[i]) or \
               (not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_WilliamsFractal_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0