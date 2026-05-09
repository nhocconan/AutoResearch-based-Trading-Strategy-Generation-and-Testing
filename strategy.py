#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA40 trend filter and volume confirmation
# Long when price breaks above 20-day high with weekly EMA40 uptrend and volume > 1.5x average
# Short when price breaks below 20-day low with weekly EMA40 downtrend and volume > 1.5x average
# Exit when price retraces to 10-day EMA or opposite Donchian channel
# Uses daily price structure for breakouts, weekly trend for filter, volume for conviction
# Designed to capture major trends while avoiding false breakouts in choppy markets
# Target: 40-80 total trades over 4 years (10-20/year) with size 0.25

name = "1d_Donchian20_1wEMA40_VolumeConfirm"
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
    
    # Calculate 1w Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channel
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Calculate 1w EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA and Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly 20-period high, EMA40 uptrend, volume confirmation
            if (close[i] > high_20w_aligned[i] and 
                ema40_1w_aligned[i] > ema40_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly 20-period low, EMA40 downtrend, volume confirmation
            elif (close[i] < low_20w_aligned[i] and 
                  ema40_1w_aligned[i] < ema40_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to 10-day EMA or breaks below weekly 20-period low
            ema10 = pd.Series(close[:i+1]).ewm(span=10, adjust=False).mean().iloc[-1]
            if (close[i] <= ema10) or (close[i] < low_20w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to 10-day EMA or breaks above weekly 20-period high
            ema10 = pd.Series(close[:i+1]).ewm(span=10, adjust=False).mean().iloc[-1]
            if (close[i] >= ema10) or (close[i] > high_20w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals