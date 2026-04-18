#!/usr/bin/env python3
"""
6h_Williams_Fractal_Breakout_WeeklyTrend
Hypothesis: Williams Fractal breakouts on 6h with weekly trend filter.
In strong weekly trends (price above/below weekly SMA50), trade breakouts of
recently confirmed fractals in the trend direction. Weekly trend reduces
whipsaw in ranging markets. Fractals provide natural support/resistance levels.
Designed for very low trade frequency (<15/year) to avoid fee decay while
capturing strong momentum moves in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
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
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA(50) for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate Williams fractals on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    # Fractals need 2 extra bars for confirmation (center bar + 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for weekly SMA
    
    for i in range(start_idx, n):
        if (np.isnan(sma_50_1w_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_sma = sma_50_1w_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        
        # Determine weekly trend: above SMA = uptrend, below SMA = downtrend
        uptrend = price > weekly_sma
        downtrend = price < weekly_sma
        
        if position == 0:
            # Long: price breaks above bullish fractal in weekly uptrend
            if not np.isnan(bull_fract) and price > bull_fract and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal in weekly downtrend
            elif not np.isnan(bear_fract) and price < bear_fract and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below bullish fractal OR weekly trend turns down
            if (not np.isnan(bull_fract) and price < bull_fract) or not uptrend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above bearish fractal OR weekly trend turns up
            if (not np.isnan(bear_fract) and price > bear_fract) or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Williams_Fractal_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0