#!/usr/bin/env python3
"""
1d_1w_Volume_Breakout_Trend_v1
1d strategy using weekly trend filter + daily price breakout with volume confirmation.
- Long: Price breaks above weekly high + volume > 2x average + weekly trend up
- Short: Price breaks below weekly low + volume > 2x average + weekly trend down
- Exit: Opposite breakout or weekly trend reversal
- Weekly trend: weekly close > weekly SMA40
Designed for ~10-25 trades/year per symbol (40-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    
    # Get weekly data for trend filter and breakout levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly SMA40 for trend filter
    sma_40_1w = pd.Series(close_1w).rolling(window=40, min_periods=40).mean().values
    
    # Weekly high and low for breakout levels
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Align weekly data to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    sma_40_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_40_1w)
    
    # Volume spike filter (2x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 20 for volume MA + 20 for SMA40 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(sma_40_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_trend_up = close_1w[-1] > sma_40_1w[-1] if len(close_1w) > 0 else False  # Use latest weekly close
        weekly_trend_down = close_1w[-1] < sma_40_1w[-1] if len(close_1w) > 0 else False
        
        # Daily breakout conditions
        breakout_up = high[i] > weekly_high_aligned[i]
        breakdown_down = low[i] < weekly_low_aligned[i]
        
        # Volume spike filter
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: weekly uptrend + breakout above weekly high + volume spike
            if weekly_trend_up and breakout_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + breakdown below weekly low + volume spike
            elif weekly_trend_down and breakdown_down and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend reversal or breakdown below weekly low
            if not weekly_trend_up or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend reversal or breakout above weekly high
            if not weekly_trend_down or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Volume_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0