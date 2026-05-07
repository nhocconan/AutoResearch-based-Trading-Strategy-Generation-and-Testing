#!/usr/bin/env python3
"""
6h_MultiTimeframe_1wTrend_1dSupport_Resistance
Hypothesis: 6h timeframe with weekly trend filter and daily support/resistance levels.
Uses weekly EMA20 for trend direction (bullish when price > EMA20, bearish when price < EMA20).
Looks for mean reversion entries at daily support/resistance levels:
- Long when price touches or crosses below daily S1 (Camarilla) in weekly uptrend
- Short when price touches or crosses above daily R1 (Camarilla) in weekly downtrend
Uses volume confirmation to avoid false breaks (6h volume > 1.5x 20-period average).
Designed for 15-35 trades/year to avoid fee drag in 6h timeframe.
Works in bull/bear via trend filter and mean reversion at key levels.
"""

name = "6h_MultiTimeframe_1wTrend_1dSupport_Resistance"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla levels (using prior daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Range and Camarilla levels from prior daily bar
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.1 * (range_1d / 12)  # R1 = C + 1.1*(H-L)/12
    s1_1d = close_1d - 1.1 * (range_1d / 12)  # S1 = C - 1.1*(H-L)/12
    
    # Align Camarilla levels to 6h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 6h volume for confirmation
    vol_6h = volume
    vol_ma20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = np.divide(vol_6h, vol_ma20_6h, out=np.zeros_like(vol_6h), where=vol_ma20_6h!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ratio_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        # Get weekly close aligned to 6h for trend comparison
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        if np.isnan(close_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price touches/crosses below S1 in weekly uptrend with volume
            if (low[i] <= s1_1d_aligned[i] and 
                close[i] > s1_1d_aligned[i] and  # reversal confirmation
                vol_ratio_6h[i] > 1.5 and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price touches/crosses above R1 in weekly downtrend with volume
            elif (high[i] >= r1_1d_aligned[i] and 
                  close[i] < r1_1d_aligned[i] and  # reversal confirmation
                  vol_ratio_6h[i] > 1.5 and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches/crosses above R1 or trend turns down
            if (high[i] >= r1_1d_aligned[i] and 
                close[i] < r1_1d_aligned[i]) or \
               not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches/crosses below S1 or trend turns up
            if (low[i] <= s1_1d_aligned[i] and 
                close[i] > s1_1d_aligned[i]) or \
               not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals