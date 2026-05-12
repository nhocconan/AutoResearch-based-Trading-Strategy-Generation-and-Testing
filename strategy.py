#!/usr/bin/env python3
name = "1d_TripleBarrier_WickReversal_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter
    df_week = get_htf_data(prices, '1w')
    close_week = df_week['close'].values
    ema20_week = pd.Series(close_week).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_week_aligned = align_htf_to_ltf(prices, df_week, ema20_week)
    
    # Daily triple barrier
    high_1d = high
    low_1d = low
    close_1d = close
    volume_1d = volume
    
    # Calculate 20-day rolling high/low for breakout
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1d > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_week_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 20-day high + weekly uptrend + volume confirmation
            if (close[i] > high_20[i] and 
                close[i] > ema20_week_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low + weekly downtrend + volume confirmation
            elif (close[i] < low_20[i] and 
                  close[i] < ema20_week_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below 20-day low or weekly trend reversal
            if (close[i] < low_20[i] or 
                close[i] < ema20_week_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above 20-day high or weekly trend reversal
            if (close[i] > high_20[i] or 
                close[i] > ema20_week_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals