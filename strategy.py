#!/usr/bin/env python3
# 6h_MarketStructure_Breakout_VolumeFilter
# Hypothesis: Combines 6h market structure (higher highs/lows) with 12h trend filter and volume confirmation.
# In bull markets: buy breaks above recent swing highs with volume.
# In bear markets: sell breaks below recent swing lows with volume.
# Uses 12h EMA for trend filter to avoid counter-trend trades.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "6h_MarketStructure_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h swing points (5-bar lookback: 2 bars each side + current)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Swing high: higher than 2 bars on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = high[i]
        # Swing low: lower than 2 bars on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = low[i]
    
    # Forward fill swing levels to maintain structure until broken
    swing_high_fill = pd.Series(swing_high).ffill().values
    swing_low_fill = pd.Series(swing_low).ffill().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(swing_high_fill[i]) or np.isnan(swing_low_fill[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above swing high + above 12h EMA + volume confirmation
            if close[i] > swing_high_fill[i] and close[i] > ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below swing low + below 12h EMA + volume confirmation
            elif close[i] < swing_low_fill[i] and close[i] < ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below swing low or goes below 12h EMA
            if close[i] < swing_low_fill[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above swing high or goes above 12h EMA
            if close[i] > swing_high_fill[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals