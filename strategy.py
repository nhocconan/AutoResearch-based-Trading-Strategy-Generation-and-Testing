#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_williams_fractal_reversal_v1
# Uses weekly Williams fractal reversal with daily trend filter on 6h timeframe.
# In bull markets, buys at bullish fractal (support) when price > daily EMA50.
# In bear markets, shorts at bearish fractal (resistance) when price < daily EMA50.
# Weekly fractals provide high-probability reversal zones, daily EMA filters counter-trend noise.
# Target: 15-25 trades/year per symbol for low friction and high edge in ranging/trending markets.

name = "6h_1w_1d_williams_fractal_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Williams fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams fractals on weekly data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-1] < low[n+1]
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    # Need at least 5 points for fractal calculation (2 left, 2 right)
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal: peak with lower highs on both sides
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and 
            high_1w[i-3] < high_1w[i-1] and 
            high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        
        # Bullish fractal: trough with higher lows on both sides
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and 
            low_1w[i-3] > low_1w[i-1] and 
            low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Weekly fractals need 2-bar confirmation after the fractal bar
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price at bullish fractal support and above daily EMA50
        if (not np.isnan(bullish_fractal_aligned[i]) and 
            close[i] <= bullish_fractal_aligned[i] * 1.005 and  # allow small tolerance
            close[i] > ema_50_aligned[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        
        # Short signal: price at bearish fractal resistance and below daily EMA50
        elif (not np.isnan(bearish_fractal_aligned[i]) and 
              close[i] >= bearish_fractal_aligned[i] * 0.995 and  # allow small tolerance
              close[i] < ema_50_aligned[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        
        # Exit conditions: opposite fractal touch or EMA cross
        elif position == 1:
            # Exit long if price touches bearish fractal or falls below EMA50
            if ((not np.isnan(bearish_fractal_aligned[i]) and 
                 close[i] >= bearish_fractal_aligned[i] * 0.995) or
                close[i] < ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # hold long
        
        elif position == -1:
            # Exit short if price touches bullish fractal or rises above EMA50
            if ((not np.isnan(bullish_fractal_aligned[i]) and 
                 close[i] <= bullish_fractal_aligned[i] * 1.005) or
                close[i] > ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # hold short
        
        else:
            signals[i] = 0.0  # remain flat
    
    return signals