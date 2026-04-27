#!/usr/bin/env python3
"""
6h_MACD_Histogram_Reversal_1dTrend_Filter
Hypothesis: MACD histogram reversal (from negative to positive for longs, positive to negative for shorts) 
on 6h timeframe captures momentum shifts. Filtered by 1d EMA50 trend to avoid counter-trend trades. 
Works in bull via MACD bullish reversals above EMA50 and bear via bearish reversals below EMA50. 
Target: ~25 trades/year on 6h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate MACD on 6h data
    # MACD line: EMA12 - EMA26
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    # Signal line: EMA9 of MACD line
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    # MACD histogram: MACD line - Signal line
    macd_hist = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for MACD (max period 26+9=35)
    start_idx = max(35, 50)
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        hist = macd_hist[i]
        hist_prev = macd_hist[i-1]
        ema_trend = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: MACD histogram crosses above zero (bullish reversal) with uptrend
            if hist > 0 and hist_prev <= 0 and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: MACD histogram crosses below zero (bearish reversal) with downtrend
            elif hist < 0 and hist_prev >= 0 and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: MACD histogram crosses below zero or trend turns down
            if hist < 0 and hist_prev >= 0 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: MACD histogram crosses above zero or trend turns up
            if hist > 0 and hist_prev <= 0 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_MACD_Histogram_Reversal_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0