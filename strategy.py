#!/usr/bin/env python3
"""
6h_TurtleChannel_1dTrendFilter_V1
Hypothesis: Turtle Trading 20-day breakout on 6h chart, filtered by 1d EMA50 trend direction, captures medium-term trends in both bull and bear markets. Uses volume confirmation to filter false breakouts. Target 20-30 trades per year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align daily EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h data for breakout calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period high/low channels (Turtle breakout)
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - 19)
        high_20[i] = np.max(high[start_idx:i+1])
        low_20[i] = np.min(low[start_idx:i+1])
    
    # Volume filter: volume > 1.3x 20-period average
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        volume_avg[i] = np.mean(volume[start_idx:i+1])
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_channel = high_20[i-1]  # Previous bar's 20-period high
        lower_channel = low_20[i-1]   # Previous bar's 20-period low
        ema50 = ema50_1d_aligned[i]
        vol_confirm = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above 20-day high channel with volume confirmation in uptrend
            if price > upper_channel and vol_confirm and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 20-day low channel with volume confirmation in downtrend
            elif price < lower_channel and vol_confirm and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to 20-day low channel or trend breaks
            if price < lower_channel or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 20-day high channel or trend breaks
            if price > upper_channel or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_TurtleChannel_1dTrendFilter_V1"
timeframe = "6h"
leverage = 1.0