#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_With_Trend_Filter
Hypothesis: Daily Camarilla pivot levels (S3/R3) provide strong support/resistance.
Breakouts above R3 or below S3 on 4h chart with trend alignment (200 EMA) and
volume expansion capture institutional moves. Works in both bull and bear markets
by trading with the dominant trend. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first bar uses its own close
    
    range_1d = high_1d - low_1d
    
    # Resistance levels (R3 used)
    R3 = close_prev + (range_1d * 1.2500 / 4)
    
    # Support levels (S3 used)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    
    # Align levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Trend filter: 200-period EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(ema_200[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R3 with volume expansion and above EMA200
        long_breakout = close[i] > R3_aligned[i] and volume_expansion[i] and close[i] > ema_200[i]
        
        # Short breakdown: price breaks below S3 with volume expansion and below EMA200
        short_breakout = close[i] < S3_aligned[i] and volume_expansion[i] and close[i] < ema_200[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0