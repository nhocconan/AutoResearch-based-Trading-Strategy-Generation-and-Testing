#!/usr/bin/env python3
"""
12h_Rolling_MaxMin_Breakout_With_Volume_Confirmation
Hypothesis: Use 12h rolling 20-period high/low as dynamic support/resistance. Go long when price breaks above 20-period high with volume confirmation, short when breaks below 20-period low. Uses 1w trend filter (price > 200-period EMA on weekly) to avoid counter-trend trades. Designed to capture momentum breaks in both bull and bear markets while minimizing false breakouts. Targets 15-25 trades/year with position size 0.25.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 200-period EMA on weekly close
    if len(close_1w) < 200:
        ema_200_1w = np.full(len(close_1w), np.nan)
    else:
        ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA to 12h timeframe
    ema_200_12h = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-period rolling high/low on 12h
    roll_high = np.full(n, np.nan)
    roll_low = np.full(n, np.nan)
    for i in range(20, n):
        roll_high[i] = np.max(high[i-20:i])
        roll_low[i] = np.min(low[i-20:i])
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need rolling high/low and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_200_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above 20-period high with volume confirmation and above weekly EMA200
            if close[i] > roll_high[i] and vol_confirmed and close[i] > ema_200_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period low with volume confirmation and below weekly EMA200
            elif close[i] < roll_low[i] and vol_confirmed and close[i] < ema_200_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below 20-period high
            if close[i] < roll_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above 20-period low
            if close[i] > roll_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Rolling_MaxMin_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0