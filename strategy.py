#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation.
# Uses Williams Fractals on 1d to identify potential reversal points, then waits for breakout
# in direction of weekly trend. Volume confirms breakout strength. Designed to work in
# both bull and bear markets by following weekly trend direction.
# Williams Fractals require 2-bar confirmation after the pattern forms.
name = "6h_WilliamsFractal_Breakout_1wTrend_Volume"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Fractals on daily: need 2-bar confirmation after pattern
    # Bearish fractal: high[i] is highest of [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Apply 2-bar confirmation delay for fractals (need 2 more daily bars after pattern)
    bearish_fractal_confirmed = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal_confirmed = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(len(bearish_fractal)):
        if bearish_fractal[i] and i + 2 < len(bearish_fractal):
            bearish_fractal_confirmed[i + 2] = True
        if bullish_fractal[i] and i + 2 < len(bullish_fractal):
            bullish_fractal_confirmed[i + 2] = True
    
    # Align fractals to 6h timeframe with proper delay
    bearish_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed.astype(float))
    bullish_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed.astype(float))
    
    # Volume spike filter: volume > 2x 50-period EMA
    vol_ema50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_spike = volume > (2.0 * vol_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_50_6h[i]) or np.isnan(bearish_6h[i]) or 
            np.isnan(bullish_6h[i]) or np.isnan(vol_ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: bullish fractal breakout with volume and above weekly EMA50
            if bullish_6h[i] > 0.5 and vol_spike[i] and price > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakdown with volume and below weekly EMA50
            elif bearish_6h[i] > 0.5 and vol_spike[i] and price < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below weekly EMA50 or opposite fractal appears
            if price < ema_50_6h[i] or bearish_6h[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above weekly EMA50 or opposite fractal appears
            if price > ema_50_6h[i] or bullish_6h[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals