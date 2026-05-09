#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume confirmation
# Long when price breaks above Donchian upper (20) with EMA200 uptrend and volume > 1.5x average
# Short when price breaks below Donchian lower (20) with EMA200 downtrend and volume > 1.5x average
# Exit when price crosses the Donchian middle (10-period midpoint)
# Designed to capture strong breakouts with trend alignment and volume confirmation
# Target: 100-180 total trades over 4 years (25-45/year) with size 0.25

name = "4h_Donchian_Breakout_1dEMA200_Volume"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (highest_high + lowest_low) / 2
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 200)  # Need enough data for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper, EMA200 uptrend, volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema200_1d_aligned[i] and  # Price above EMA200 (uptrend)
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower, EMA200 downtrend, volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema200_1d_aligned[i] and  # Price below EMA200 (downtrend)
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses Donchian middle (mean reversion)
            if close[i] <= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses Donchian middle (mean reversion)
            if close[i] >= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals