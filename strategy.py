#!/usr/bin/env python3
name = "6h_TurtleTrader_10_40_ATR_VolumeFilter"
timeframe = "6h"
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
    
    # Load daily data for 10-period high/low breakout and 40-period exit
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 10-day high and low (Turtle entry)
    high_10d = pd.Series(df_1d['high']).rolling(window=10, min_periods=10).max().values
    low_10d = pd.Series(df_1d['low']).rolling(window=10, min_periods=10).min().values
    
    # Calculate 40-day high and low (Turtle exit)
    high_40d = pd.Series(df_1d['high']).rolling(window=40, min_periods=40).max().values
    low_40d = pd.Series(df_1d['low']).rolling(window=40, min_periods=40).min().values
    
    # Align to 6h timeframe (wait for daily close)
    high_10d_aligned = align_htf_to_ltf(prices, df_1d, high_10d)
    low_10d_aligned = align_htf_to_ltf(prices, df_1d, low_10d)
    high_40d_aligned = align_htf_to_ltf(prices, df_1d, high_40d)
    low_40d_aligned = align_htf_to_ltf(prices, df_1d, low_40d)
    
    # ATR for volatility filter (using daily data)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_10d_aligned[i]) or np.isnan(low_10d_aligned[i]) or 
            np.isnan(high_40d_aligned[i]) or np.isnan(low_40d_aligned[i]) or
            np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if volatility too low (ATR < 0.5% of price)
        if atr_aligned[i] < 0.005 * close[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 10-day high + volume filter
            if close[i] > high_10d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 10-day low + volume filter
            elif close[i] < low_10d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 40-day low
            if close[i] < low_40d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 40-day high
            if close[i] > high_40d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals