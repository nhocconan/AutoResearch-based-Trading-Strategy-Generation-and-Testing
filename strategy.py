#!/usr/bin/env python3
"""
12h_supertrend_weekly_filter_v1
Hypothesis: Supertrend on 12h with weekly trend filter to avoid counter-trend trades.
- Long when Supertrend turns green and weekly trend is bullish
- Short when Supertrend turns red and weekly trend is bearish
- Exit when Supertrend reverses
- Targets 15-30 trades/year to minimize fee decay
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_supertrend_weekly_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    atr[atr_period:] = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values[atr_period:]
    
    # Calculate upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
        
        # Prevent reversal
        if direction[i] == 1 and supertrend[i] < supertrend[i-1]:
            supertrend[i] = supertrend[i-1]
        if direction[i] == -1 and supertrend[i] > supertrend[i-1]:
            supertrend[i] = supertrend[i-1]
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA for trend
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_bullish = close_1w > ema_21
    weekly_bearish = close_1w < ema_21
    
    # Align weekly trend to 12h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(atr_period, 21)  # Wait for indicators to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Supertrend turns red or weekly turns bearish
            if direction[i] == -1 or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Supertrend turns green or weekly turns bullish
            if direction[i] == 1 or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Supertrend turns green and weekly bullish
            if direction[i] == 1 and direction[i-1] == -1 and weekly_bullish_aligned[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Short entry: Supertrend turns red and weekly bearish
            elif direction[i] == -1 and direction[i-1] == 1 and weekly_bearish_aligned[i] > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals