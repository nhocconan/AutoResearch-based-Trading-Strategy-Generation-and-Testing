#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Fractal breakout with weekly trend filter and volume confirmation.
Long when price breaks above recent bullish fractal resistance, weekly close > weekly open (bullish), and volume > 2x average.
Short when price breaks below recent bearish fractal support, weekly close < weekly open (bearish), and volume > 2x average.
Exit when price returns to the opposite fractal level or volume drops below average.
Williams Fractals identify key support/resistance levels; weekly trend filters ensure alignment with higher timeframe momentum.
Designed for low trade frequency (~15-30/year) to capture breakouts with strong institutional participation.
Works in bull markets via breakouts and in bear markets via breakdowns, both requiring volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (up) and bullish (down)"""
    n = len(high)
    bearish = np.zeros(n, dtype=bool)  # peak: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n] < high[n-1]
    bullish = np.zeros(n, dtype=bool)  # trough: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n] > low[n-1]
    
    for i in range(2, n-2):
        # Bearish fractal (peak)
        if (high[i-2] < high[i-1] and high[i] < high[i-1] and 
            high[i-3] < high[i-2] and high[i+1] < high[i-1]):
            bearish[i] = True
        # Bullish fractal (trough)
        if (low[i-2] > low[i-1] and low[i] > low[i-1] and 
            low[i-3] > low[i-2] and low[i+1] > low[i-1]):
            bullish[i] = True
            
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly trend: bullish if close > open, bearish if close < open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bearish = weekly_close < weekly_open  # True for bearish week
    
    # Calculate Williams Fractals on 6h data
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high, low)
    
    # Find most recent fractal levels
    recent_bearish = np.full(n, np.nan)  # most recent bearish fractal high (resistance for shorts)
    recent_bullish = np.full(n, np.nan)  # most recent bullish fractal low (support for longs)
    
    last_bearish_idx = -1
    last_bullish_idx = -1
    
    for i in range(n):
        if bearish_fractal[i]:
            last_bearish_idx = i
        if bullish_fractal[i]:
            last_bullish_idx = i
            
        if last_bearish_idx != -1:
            recent_bearish[i] = high[last_bearish_idx]
        if last_bullish_idx != -1:
            recent_bullish[i] = low[last_bullish_idx]
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(recent_bearish[i]) or np.isnan(recent_bullish[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above recent bullish fractal resistance, weekly bullish, volume surge
            if (recent_bullish[i] > 0 and close[i] > recent_bullish[i] and 
                weekly_bull and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent bearish fractal support, weekly bearish, volume surge
            elif (recent_bearish[i] > 0 and close[i] < recent_bearish[i] and 
                  weekly_bear and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to recent bullish fractal support OR weekly turns bearish
                if (recent_bullish[i] > 0 and close[i] < recent_bullish[i]) or weekly_bear:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to recent bearish fractal resistance OR weekly turns bullish
                if (recent_bearish[i] > 0 and close[i] > recent_bearish[i]) or weekly_bull:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractal_1wTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0