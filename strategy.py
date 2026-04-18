#!/usr/bin/env python3
"""
6h_Liquidity_Pullback_Strategy
Hypothesis: In 6h timeframe, price often pulls back to liquidity pools (equal highs/lows) before continuing the trend.
Enter long when: price pulls back to recent equal low, closes above it, and 1d trend is up (close > EMA50).
Enter short when: price pulls back to recent equal high, closes below it, and 1d trend is down (close < EMA50).
Use volume confirmation (1.5x average) to ensure genuine interest.
Targets 15-25 trades/year by requiring both liquidity touch and trend alignment.
Works in bull/bear by following higher timeframe trend while exploiting mean reversion to liquidity.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * 0.04 + ema50_1d[i-1] * 0.96
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Find equal highs/lows (liquidity pools) - look back 20 bars
    equal_high = np.full(n, np.nan)
    equal_low = np.full(n, np.nan)
    lookback = 20
    
    for i in range(lookback, n):
        # Check for equal high (within 0.1% tolerance)
        high_window = high[i-lookback:i]
        if np.max(high_window) - np.min(high_window) <= 0.001 * np.max(high_window):
            equal_high[i] = np.max(high_window)
        
        # Check for equal low (within 0.1% tolerance)
        low_window = low[i-lookback:i]
        if np.max(low_window) - np.min(low_window) <= 0.001 * np.min(low_window):
            equal_low[i] = np.min(low_window)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: pullback to equal low + close above it + uptrend + volume
            if (not np.isnan(equal_low[i]) and 
                close[i] > equal_low[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: pullback to equal high + close below it + downtrend + volume
            elif (not np.isnan(equal_high[i]) and 
                  close[i] < equal_high[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below equal low or trend changes
            if (not np.isnan(equal_low[i]) and close[i] < equal_low[i]) or close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above equal high or trend changes
            if (not np.isnan(equal_high[i]) and close[i] > equal_high[i]) or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Liquidity_Pullback_Strategy"
timeframe = "6h"
leverage = 1.0