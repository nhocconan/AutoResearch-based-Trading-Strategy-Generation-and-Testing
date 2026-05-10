#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: 12h Camarilla R1/S1 breakout in direction of 1d EMA34 trend, with volume confirmation. Camarilla levels act as intraday support/resistance, breakouts capture momentum. Works in both bull and bear by following higher timeframe trend. Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get price, volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current 12h bar using previous bar's OHLC
        if i == 0:
            # Not enough history for previous bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Previous bar's high, low, close
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        # Camarilla levels
        range_val = phigh - plow
        if range_val <= 0:
            # Avoid division by zero or invalid range
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # R1 and S1 levels (most important for intraday trading)
        r1 = pclose + (range_val * 1.1 / 12)
        s1 = pclose - (range_val * 1.1 / 12)
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above R1 with volume
            if close[i] > ema_34_aligned[i] and high[i] > r1 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below S1 with volume
            elif close[i] < ema_34_aligned[i] and low[i] < s1 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR trend turns bearish
            if low[i] < s1 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR trend turns bullish
            if high[i] > r1 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals