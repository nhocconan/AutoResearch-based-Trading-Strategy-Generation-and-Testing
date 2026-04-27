#!/usr/bin/env python3
"""
#100948 - 12h_WeeklyHighLow_Breakout_1wTrend_Volume
Hypothesis: Breakout above weekly high or below weekly low with weekly trend filter and volume confirmation on 12h timeframe. Uses weekly high/low as dynamic support/resistance levels. Works in trending markets (breakouts continue trend) and ranging markets (mean reversion to weekly range). Weekly trend filter avoids counter-trend trades. Volume confirmation ensures breakout legitimacy. Targets 15-35 trades/year to minimize fee drag.
"""

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
    
    # Get weekly data for trend filter and weekly high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate weekly high and low (using previous week's values to avoid look-ahead)
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Align weekly high/low to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume filter: volume > 2.0x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above weekly high, above weekly EMA20, volume spike
        if (close[i] > weekly_high_aligned[i] and 
            close[i] > ema20_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below weekly low, below weekly EMA20, volume spike
        elif (close[i] < weekly_low_aligned[i] and 
              close[i] < ema20_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly midpoint (mean reversion to weekly range)
        elif position == 1 and close[i] < (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyHighLow_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0