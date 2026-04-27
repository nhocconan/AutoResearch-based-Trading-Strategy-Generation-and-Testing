#!/usr/bin/env python3
"""
#100946 - 4h_Donchian20_Breakout_Volume_Trend_SR
Hypothesis: Donchian(20) breakout with volume confirmation and 1d EMA trend filter on 4h timeframe. 
Targets 20-50 trades/year to minimize fee drag. Uses 4h primary timeframe with 1d EMA50 for trend 
filter and Donchian channels for breakout. Works in bull (breakouts with trend) and bear 
(mean reversion to middle band). Tight entry conditions reduce trade frequency and improve win rate.
"""

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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_channel = (highest_high + lowest_low) / 2
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian band, above 1d EMA50, volume spike
        if (close[i] > highest_high[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian band, below 1d EMA50, volume spike
        elif (close[i] < lowest_low[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle Donchian band (mean reversion)
        elif position == 1 and close[i] < middle_channel[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > middle_channel[i]:
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

name = "4h_Donchian20_Breakout_Volume_Trend_SR"
timeframe = "4h"
leverage = 1.0