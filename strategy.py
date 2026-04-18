#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: On 6h timeframe, price breaks Donchian(20) channels with volume confirmation,
filtered by weekly pivot direction (bullish/bearish based on weekly pivot point).
This combines price breakout structure with institutional pivot levels to filter false breakouts,
working in both bull and bear markets by aligning with weekly bias. Targets 15-30 trades/year.
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
    
    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point and support/resistance levels
    pivot = np.full_like(high_1w, np.nan)
    r1 = np.full_like(high_1w, np.nan)
    s1 = np.full_like(high_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 1:  # Need previous week data
            prev_high = high_1w[i-1]
            prev_low = low_1w[i-1]
            prev_close = close_1w[i-1]
            pivot[i] = (prev_high + prev_low + prev_close) / 3.0
            r1[i] = 2 * pivot[i] - prev_low
            s1[i] = 2 * pivot[i] - prev_high
    
    # Calculate Donchian channel (20-period) on 6h data
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    lookback = 20
    
    for i in range(lookback, len(high)):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align weekly pivot data to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(pivot_6h[i]) or 
            np.isnan(r1_6h[i]) or np.isnan(s1_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Weekly bias: above pivot = bullish bias, below pivot = bearish bias
        bullish_bias = close[i] > pivot_6h[i]
        bearish_bias = close[i] < pivot_6h[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and bullish bias
            if close[i] > donchian_high[i] and vol_confirm and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and bearish bias
            elif close[i] < donchian_low[i] and vol_confirm and bearish_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR weekly bias turns bearish
            if close[i] < donchian_low[i] or not bullish_bias:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR weekly bias turns bullish
            if close[i] > donchian_high[i] or bullish_bias:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0