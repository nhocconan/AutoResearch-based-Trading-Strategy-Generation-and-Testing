#!/usr/bin/env python3
"""
4h_Price_Action_With_Volume_and_Trend_Filter
Hypothesis: Use 4h price action relative to 4h 20-period EMA (trend) and volume confirmation to capture momentum moves. 
Go long when price closes above EMA20 with above-average volume, short when price closes below EMA20 with above-average volume.
Uses 1D trend filter (price above/below 1D EMA50) to avoid counter-trend trades in strong trends. 
Designed to work in both bull and bear markets by aligning with higher timeframe trend. 
Targets 20-30 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA20 for trend
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1D EMA50 for trend filter
    close_1d_s = pd.Series(close_1d)
    ema50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # need EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        vol_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        if position == 0:
            # Long entry: price closes above EMA20 with volume confirmation and 1D uptrend
            if close[i] > ema20[i] and vol_confirmed and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price closes below EMA20 with volume confirmation and 1D downtrend
            elif close[i] < ema20[i] and vol_confirmed and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price closes below EMA20
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Price_Action_With_Volume_and_Trend_Filter"
timeframe = "4h"
leverage = 1.0