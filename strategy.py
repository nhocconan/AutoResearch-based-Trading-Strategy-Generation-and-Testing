#!/usr/bin/env python3
# 6H_Volume_Weighted_Momentum_1DTrend_Filter
# Hypothesis: Uses volume-weighted momentum (VWAP deviation) on 6h timeframe filtered by 1-day trend.
# Long when price > VWAP and 1d trend up; short when price < VWAP and 1d trend down.
# Volume confirmation ensures momentum is genuine. Works in bull/bear by following 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6H_Volume_Weighted_Momentum_1DTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for 6h (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # 1-day EMA50 for trend filter (more responsive than 34 for 6h signals)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x average volume (50-period to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have VWAP and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price above VWAP + Uptrend (price > EMA50) + volume confirmation
            if (close[i] > vwap[i] and 
                close[i] > ema50_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below VWAP + Downtrend (price < EMA50) + volume confirmation
            elif (close[i] < vwap[i] and 
                  close[i] < ema50_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls back below VWAP or trend turns down
            if close[i] < vwap[i] or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above VWAP or trend turns up
            if close[i] > vwap[i] or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals