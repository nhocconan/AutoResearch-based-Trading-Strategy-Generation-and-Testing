#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation
# Uses Donchian channel breakouts for trend following, filtered by daily EMA to ensure
# alignment with higher timeframe trend. Volume confirms breakout strength.
# Works in bull markets (long breakouts above EMA) and bear markets (short breakdowns below EMA).
# Target: 50-150 total trades over 4 years (12-38/year). Focus on BTC/ETH as primary assets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 4h timeframe (waits for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian(20) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size as fraction of capital
    
    for i in range(20, n):
        # Skip if EMA data not available
        if np.isnan(ema_50_aligned[i]):
            continue
        
        # Long entry: price breaks above Donchian upper band + price above daily EMA + volume confirmation
        if (high[i] > highest_high[i] and  # Current high breaks upper band
            close[i] > ema_50_aligned[i] and  # Price above daily EMA
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # Volume spike
            position <= 0):  # Not already long
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower band + price below daily EMA + volume confirmation
        elif (low[i] < lowest_low[i] and  # Current low breaks lower band
              close[i] < ema_50_aligned[i] and  # Price below daily EMA
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # Volume spike
              position >= 0):  # Not already short
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout or price crosses EMA in opposite direction
        elif position == 1 and (low[i] < lowest_low[i] or close[i] < ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (high[i] > highest_high[i] or close[i] > ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_EMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0