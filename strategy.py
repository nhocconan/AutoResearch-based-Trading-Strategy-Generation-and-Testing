#!/usr/bin/env python3
# 4h_Donchian_20_Breakout_Volume_Trend_Strategy
# Hypothesis: 4-hour Donchian channel breakouts with volume confirmation and trend filter
# Donchian(20) provides clear breakout levels from recent price action
# Volume > 1.5x 20-period average confirms institutional participation
# Trend filter uses 4h EMA50 to avoid counter-trend trades
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# Works in bull/bear via trend filter - only trades in direction of EMA50 trend

name = "4h_Donchian_20_Breakout_Volume_Trend_Strategy"
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
    
    # 4h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channel (20-period) - highest high and lowest low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA50 for long, below for short
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and uptrend
            if (close[i] > donchian_high[i] and 
                volume_confirm[i] and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_confirm[i] and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or trend changes
            if (close[i] < donchian_low[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or trend changes
            if (close[i] > donchian_high[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals