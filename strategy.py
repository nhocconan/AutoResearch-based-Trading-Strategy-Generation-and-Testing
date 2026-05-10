#!/usr/bin/env python3
# 6h_RSI_Streak_With_Trend_and_Volume
# Hypothesis: RSI streak (consecutive closes above/below 50) identifies momentum bursts.
# In trending markets (1d EMA50), streaks of 3+ indicate strong directional moves.
# Volume confirmation filters false signals. Works in bull/bear by following daily trend.

name = "6h_RSI_Streak_With_Trend_and_Volume"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate RSI streak: consecutive closes above/below 50
    above_50 = rsi > 50
    below_50 = rsi < 50
    
    # Streak of consecutive True values
    streak_above = np.zeros(n, dtype=int)
    streak_below = np.zeros(n, dtype=int)
    
    for i in range(1, n):
        if above_50[i]:
            streak_above[i] = streak_above[i-1] + 1
        else:
            streak_above[i] = 0
            
        if below_50[i]:
            streak_below[i] = streak_below[i-1] + 1
        else:
            streak_below[i] = 0
    
    # Volume confirmation (20-period MA on 6h = ~5 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA50 (50), RSI (14), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + RSI streak >= 3 + volume
            if uptrend and streak_above[i] >= 3 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + RSI streak <= -3 (streak_below >= 3) + volume
            elif downtrend and streak_below[i] >= 3 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or RSI streak breaks
            if not uptrend or streak_above[i] == 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or RSI streak breaks
            if not downtrend or streak_below[i] == 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals