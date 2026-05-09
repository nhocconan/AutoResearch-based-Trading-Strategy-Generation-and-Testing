#!/usr/bin/env python3
# 1D_1W_RandomForestBreakout_TrendVolume
# Hypothesis: On 1d timeframe, enter long when price breaks above weekly 200-day high with weekly uptrend and volume confirmation.
# Short when price breaks below weekly 200-day low with weekly downtrend and volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades and weekly high/low levels for precise entries.
# Target: 10-30 trades/year per symbol (40-120 total over 4 years).

name = "1D_1W_RandomForestBreakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly 200-period high and low for breakout levels
    high_200 = pd.Series(high_1w).rolling(window=200, min_periods=200).max().values
    low_200 = pd.Series(low_1w).rolling(window=200, min_periods=200).min().values
    
    # Weekly trend: price above/below 50-period EMA
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1w > ema_50
    
    # Volume confirmation: current daily volume > 1.5x 20-day average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly indicators to daily
    high_200_aligned = align_htf_to_ltf(prices, df_1w, high_200)
    low_200_aligned = align_htf_to_ltf(prices, df_1w, low_200)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(high_200_aligned[i]) or np.isnan(low_200_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly 200-day high + weekly uptrend + volume confirmation
            if close[i] > high_200_aligned[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly 200-day low + weekly downtrend + volume confirmation
            elif close[i] < low_200_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly 200-day low or trend changes
            if close[i] < low_200_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly 200-day high or trend changes
            if close[i] > high_200_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals