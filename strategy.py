#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# Long when price breaks above 20-period high with 12h EMA50 uptrend and volume > 1.5x average
# Short when price breaks below 20-period low with 12h EMA50 downtrend and volume > 1.5x average
# Exit when price retraces to 10-period EMA on 6h timeframe
# Uses Donchian channels for breakout detection, EMA for trend, volume for conviction
# Designed to capture momentum in both trending and ranging markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_Donchian_Breakout_12hEMA50_Volume"
timeframe = "6h"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 6-period EMA for exit (faster response)
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Calculate Donchian channels (20-period)
    # Using rolling window on high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(ema6[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 12h EMA50 uptrend, volume confirmation
            if (close[i] > donchian_high[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 12h EMA50 downtrend, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to 6-period EMA
            if close[i] <= ema6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to 6-period EMA
            if close[i] >= ema6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals