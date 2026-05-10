#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_1dTrend_Volume
Hypothesis: Price breaks Camarilla R1 (resistance) or S1 (support) levels calculated from the prior day,
with volume confirmation and 1d trend filter. Uses 1d EMA50 as trend filter and 1d volume spike
for confirmation. Designed for 12h timeframe to capture multi-day moves while minimizing trades.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

name = "12h_Camarilla_R1_S1_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1)
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 12h timeframe (with 1-bar delay for completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume for volume confirmation (current day's volume > 1.5x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 12h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA50
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        # Volume filter: current 12h volume > 1.5x 20-day 1h volume average
        # Note: Using 12h volume compared to 1d volume MA - we need to align properly
        # For volume confirmation, we'll use current 12h volume vs previous day's volume MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume
            if high[i] > r1_aligned[i] and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume
            elif low[i] < s1_aligned[i] and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below R1 or trend fails
            if close[i] < r1_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above S1 or trend fails
            if close[i] > s1_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals