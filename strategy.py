#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Camarilla pivot levels from previous 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Calculate pivots for each 1d bar
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        R1[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 12
        S1[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 12
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: 4h volume > 1.5x 20-period SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 + 1d uptrend + volume confirmation
            if (close[i] > R1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_sma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + 1d downtrend + volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_sma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when close crosses below S1 or 1d trend turns down
            if (close[i] < S1_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when close crosses above R1 or 1d trend turns up
            if (close[i] > R1_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals