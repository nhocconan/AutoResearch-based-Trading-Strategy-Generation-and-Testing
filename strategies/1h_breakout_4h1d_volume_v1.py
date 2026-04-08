#!/usr/bin/env python3
"""
1h_breakout_4h1d_volume_v1
Hypothesis: Breakout above 4h high/low with volume confirmation and daily trend filter captures momentum moves.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by trading with daily trend.
Targets 15-37 trades/year by requiring 4h breakout + volume spike + daily trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_breakout_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for breakout levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h high/low for breakout levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h breakout levels (previous bar's high/low)
    # For breakout, we use the previous completed 4h bar's high/low
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    # First value will be incorrect due to roll, but will be handled by alignment
    
    # Align 4h levels to 1h timeframe
    high_breakout = align_htf_to_ltf(prices, df_4h, high_4h_prev)
    low_breakout = align_htf_to_ltf(prices, df_4h, low_4h_prev)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_1h[i]) or 
            np.isnan(high_breakout[i]) or 
            np.isnan(low_breakout[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h low OR trend turns down
            if close[i] < low_breakout[i] or close[i] < ema50_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price breaks above 4h high OR trend turns up
            if close[i] > high_breakout[i] or close[i] > ema50_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price breaks above 4h high + volume + uptrend
            if (close[i] > high_breakout[i] and 
                vol_confirm and 
                close[i] > ema50_1h[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below 4h low + volume + downtrend
            elif (close[i] < low_breakout[i] and 
                  vol_confirm and 
                  close[i] < ema50_1h[i]):
                position = -1
                signals[i] = -0.20
    
    return signals