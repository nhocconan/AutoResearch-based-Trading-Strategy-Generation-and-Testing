#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high with weekly uptrend and volume > 1.5x average
# Short when price breaks below 20-day low with weekly downtrend and volume > 1.5x average
# Exit when price retraces to 10-day moving average or reverses direction
# Uses Donchian channels for breakout, EMA for trend, volume for conviction
# Designed to capture sustained moves in both bull and bear markets with low frequency
# Target: 30-80 total trades over 4 years (7-20/year) with size 0.25

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema10[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high, weekly uptrend, volume confirmation
            if (close[i] > high_20[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # Weekly EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, weekly downtrend, volume confirmation
            elif (close[i] < low_20[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # Weekly EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to 10-day EMA or weekly trend turns down
            if (close[i] <= ema10[i]) or (ema50_1w_aligned[i] < ema50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to 10-day EMA or weekly trend turns up
            if (close[i] >= ema10[i]) or (ema50_1w_aligned[i] > ema50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals