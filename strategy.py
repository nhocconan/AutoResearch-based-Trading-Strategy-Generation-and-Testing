#!/usr/bin/env python3
"""
6h_Liquidity_Imbalance_Momentum
Hypothesis: On 6h timeframe, price reacts to intraday liquidity imbalances (order book pressure) that persist due to delayed institutional execution.
Buy when price breaks above prior 6h high with expanding volume and bullish imbalance; sell when breaks below prior low with bearish imbalance.
Uses 1d trend filter to avoid counter-trend trades. Works in bull (momentum continuation) and bear (mean reversion of overextended moves) regimes.
Designed for low trade frequency (~20-40/year) to minimize fee drag.
"""

name = "6h_Liquidity_Imbalance_Momentum"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 for trend filter
    ema20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema20_1d[19] = np.mean(close_1d[0:20])
        for i in range(20, len(close_1d)):
            ema20_1d[i] = (close_1d[i] * 2 + ema20_1d[i-1] * 18) / 20
    
    # Align 1d EMA to 6h
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 6-period volume average for imbalance detection
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 6:
        vol_ma[5] = np.mean(volume[0:6])
        for i in range(6, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 5 + volume[i]) / 6
    
    # Volume spike detector (current > 1.5x average)
    volume_spike = np.zeros(n, dtype=bool)
    valid_vol = ~np.isnan(vol_ma) & (vol_ma > 0)
    volume_spike[valid_vol] = volume[valid_vol] > (vol_ma[valid_vol] * 1.5)
    
    # Track 6-period high/low for breakout levels
    highest_6h = np.full_like(high, np.nan)
    lowest_6h = np.full_like(low, np.nan)
    
    if len(high) >= 6:
        for i in range(6, len(high)):
            highest_6h[i] = np.max(high[i-6:i])
            lowest_6h[i] = np.min(low[i-6:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 6)  # Need 1d EMA and 6-period lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(highest_6h[i]) or 
            np.isnan(lowest_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema20_1d_aligned[i]
        
        if position == 0:
            # Enter long: bullish trend + break above 6h high + volume spike
            if trend_up and high[i] > highest_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + break below 6h low + volume spike
            elif not trend_up and low[i] < lowest_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend turns bearish OR price breaks below 6h low
            if not trend_up or low[i] < lowest_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend turns bullish OR price breaks above 6h high
            if trend_up or high[i] > highest_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals