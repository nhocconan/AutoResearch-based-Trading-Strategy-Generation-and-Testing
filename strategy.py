#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA20 trend filter and volume confirmation
# Long when price breaks above 20-period high with EMA20 uptrend and volume > 1.5x average
# Short when price breaks below 20-period low with EMA20 downtrend and volume > 1.5x average
# Exit when price retraces to 10-period SMA in opposite direction
# Uses Donchian channels for breakout, EMA for trend filter, volume for conviction
# Designed to capture trending moves with controlled frequency to avoid overtrading
# Target: 100-180 total trades over 4 years (25-45/year) with size 0.25

name = "4h_Donchian_Breakout_12hEMA20_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA20 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    ema20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period SMA for exit
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(sma10[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, EMA20 uptrend, volume confirmation
            if (close[i] > donchian_high[i] and 
                ema20_12h_aligned[i] > ema20_12h_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, EMA20 downtrend, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema20_12h_aligned[i] < ema20_12h_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to 10-period SMA
            if close[i] < sma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to 10-period SMA
            if close[i] > sma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals