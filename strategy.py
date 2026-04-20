#!/usr/bin/env python3
# 12h_1d_PriceActionBreakout_Volume
# Hypothesis: Breakout strategy using 12h price action combined with 1d trend filter and volume confirmation.
# Targets 20-30 trades/year by requiring clean breakouts from 12h swing points with volume surge.
# Works in bull markets (breakout continuation) and bear markets (mean reversion from overextended levels).
# Uses 1d EMA50 for trend filter to avoid counter-trend trades in strong trends.

name = "12h_1d_PriceActionBreakout_Volume"
timeframe = "12h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h swing high/low (3-bar lookback)
    swing_high = np.maximum.reduce([
        high,
        np.roll(high, 1),
        np.roll(high, 2)
    ])
    swing_low = np.minimum.reduce([
        low,
        np.roll(low, 1),
        np.roll(low, 2)
    ])
    
    # Align swing levels (already on 12h timeframe, but ensure alignment)
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)  # Using 1d as reference for alignment
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 12h swing high with volume surge and above 1d EMA50
            if (close[i] > swing_high_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 12h swing low with volume surge and below 1d EMA50
            elif (close[i] < swing_low_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below 12h swing low
            if close[i] < swing_low_aligned[i] * 0.997:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above 12h swing high
            if close[i] > swing_high_aligned[i] * 1.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals