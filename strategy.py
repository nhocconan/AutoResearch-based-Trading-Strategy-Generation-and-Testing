#!/usr/bin/env python3
"""
4H_Donchian_20_VolumeTrend_200EMA
Hypothesis: 4h price breaks above/below Donchian(20) channels with volume confirmation and 200EMA trend filter.
Works in bull/bear markets: Breakouts capture strong momentum, volume validates strength, 200EMA ensures trend alignment.
Targets 20-50 trades/year to minimize fee drag on 4h timeframe.
"""
name = "4H_Donchian_20_VolumeTrend_200EMA"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 200EMA for trend filter (using 4h data)
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 4h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(200, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_200[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 60 bars between trades (10 days on 4h TF) to reduce frequency
            if bars_since_exit < 60:
                continue
                
            # Long: price breaks above upper Donchian with volume and above 200EMA
            if (close[i] > upper[i] and close[i-1] <= upper[i-1] and 
                close[i] > ema_200[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below lower Donchian with volume and below 200EMA
            elif (close[i] < lower[i] and close[i-1] >= lower[i-1] and 
                  close[i] < ema_200[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian band (mean reversion)
            if position == 1 and close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals