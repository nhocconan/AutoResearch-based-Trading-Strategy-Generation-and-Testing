#!/usr/bin/env python3
"""
12h_MACD_Histogram_Divergence_1dTrend_Filter_V1
Hypothesis: MACD histogram divergence on 12h combined with 1d EMA trend filter provides high-probability reversals in both bull and bear markets. The 1d EMA filter ensures trades align with higher timeframe trend, reducing whipsaw. MACD histogram divergence captures momentum exhaustion before price reversals. Works in bull markets (buy bullish divergence in uptrend) and bear markets (sell bearish divergence in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 12h data for MACD
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Calculate MACD on 12h data
    macd_line, signal_line, histogram = calculate_macd(close_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(histogram[i]) or np.isnan(histogram[i-1]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        ema_trend = ema_20_1d_aligned[i]
        hist = histogram[i]
        hist_prev = histogram[i-1]
        
        # Determine trend: above EMA20 = uptrend, below = downtrend
        is_uptrend = price > ema_trend
        is_downtrend = price < ema_trend
        
        # Bullish divergence: price makes lower low, MACD histogram makes higher low
        bearish_divergence = False
        bullish_divergence = False
        
        # Look for divergence over last 5 bars
        lookback = 5
        start_idx = max(0, i - lookback)
        
        if i >= lookback:
            # Find local lows in price and histogram
            price_slice = price[start_idx:i+1]
            hist_slice = hist[start_idx:i+1]
            
            # Simple local min/max detection
            price_min_idx = np.argmin(price_slice)
            hist_min_idx = np.argmin(hist_slice)
            price_max_idx = np.argmax(price_slice)
            hist_max_idx = np.argmax(hist_slice)
            
            # Bullish divergence: price makes lower low, histogram makes higher low
            if (price_min_idx == lookback and  # recent price low
                hist_min_idx < lookback and    # earlier histogram low
                price[i] < price[i-1] and      # price making lower low
                hist > hist_prev):             # histogram making higher low
                bullish_divergence = True
            
            # Bearish divergence: price makes higher high, histogram makes lower high
            if (price_max_idx == lookback and  # recent price high
                hist_max_idx < lookback and    # earlier histogram high
                price[i] > price[i-1] and      # price making higher high
                hist < hist_prev):             # histogram making lower high
                bearish_divergence = True
        
        if position == 0:
            # Long: bullish divergence + uptrend
            if bullish_divergence and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + downtrend
            elif bearish_divergence and is_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA20 or bearish divergence
            if price < ema_trend or bearish_divergence:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 or bullish divergence
            if price > ema_trend or bullish_divergence:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_MACD_Histogram_Divergence_1dTrend_Filter_V1"
timeframe = "12h"
leverage = 1.0