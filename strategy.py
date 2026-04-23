#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
- Primary timeframe: 12h (low frequency to minimize fee drag)
- Entry: Price breaks above/below 20-period Donchian channel + trend alignment (1d EMA50) + volume > 1.5x 20-period MA
- Exit: Price retracement to midpoint of Donchian channel or trend reversal
- Designed for 12-37 trades/year per symbol to avoid fee drag while capturing medium-term trends
- Works in both bull and bear markets via 1d EMA50 trend filter and volume confirmation
"""

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels from 1d data
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: 20-period high
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Midpoint: average of upper and lower
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    midpoint_20_aligned = align_htf_to_ltf(prices, df_1d, midpoint_20)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # need EMA50_1d, Donchian(20), vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(midpoint_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Upper Donchian (breakout) AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower Donchian (breakdown) AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close retracement to midpoint OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < midpoint OR price < 1d EMA50
                if close[i] < midpoint_20_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > midpoint OR price > 1d EMA50
                if close[i] > midpoint_20_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0