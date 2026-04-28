#!/usr/bin/env python3
"""
4h_InsideBar_Breakout_1dTrend_VolumeFilter
Hypothesis: Inside bars on 4h indicate consolidation. Breakouts of inside bar high/low with 1-day EMA trend filter and volume spike capture explosive moves in both bull and bear markets. Inside bar reduces false breakouts, and volume filter ensures momentum. Targets 20-40 trades/year.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Inside bar detection: current high < previous high AND current low > previous low
    inside_bar = (high < np.roll(high, 1)) & (low > np.roll(low, 1))
    
    # Inside bar high and low (use previous bar's high/low as the inside bar boundaries)
    inside_high = np.roll(high, 1)
    inside_low = np.roll(low, 1)
    
    # Align inside bar levels to current timeframe (they are already 4h, no need to align from HTF)
    # But we need to ensure we use the inside bar from the bar that just closed
    inside_high_aligned = inside_high  # already at 4h resolution
    inside_low_aligned = inside_low
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(inside_high_aligned[i]) or
            np.isnan(inside_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions: break of inside bar high/low
        breakout_high = close[i] > inside_high_aligned[i]
        breakdown_low = close[i] < inside_low_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_high
        short_entry = vol_confirm and downtrend and breakdown_low
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakdown_low or (not uptrend)
        short_exit = breakout_high or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_InsideBar_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0