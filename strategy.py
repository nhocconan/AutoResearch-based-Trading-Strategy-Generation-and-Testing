#!/usr/bin/env python3
"""
1d_breakout_1w_trend_volume_v1
Hypothesis: Daily breakouts above weekly high/low with volume confirmation and weekly trend filter capture momentum moves.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by trading with weekly trend.
Targets 10-25 trades/year by requiring daily breakout + volume spike + weekly trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly high/low for breakout levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly levels to daily timeframe
    high_breakout = align_htf_to_ltf(prices, df_1w, high_1w)
    low_breakout = align_htf_to_ltf(prices, df_1w, low_1w)
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d[i]) or 
            np.isnan(high_breakout[i]) or 
            np.isnan(low_breakout[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.5x average volume
        vol_confirm = volume[i] > 2.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly low OR trend turns down
            if close[i] < low_breakout[i] or close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above weekly high OR trend turns up
            if close[i] > high_breakout[i] or close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly high + volume + uptrend
            if (close[i] > high_breakout[i] and 
                vol_confirm and 
                close[i] > ema50_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly low + volume + downtrend
            elif (close[i] < low_breakout[i] and 
                  vol_confirm and 
                  close[i] < ema50_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals