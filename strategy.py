#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike.
- Primary timeframe: 12h, HTF: 1w for EMA50 trend alignment.
- Donchian channel from prior 20 periods (20*12h = 10d): long at upper breakout, short at lower breakdown.
- Trend filter: only long when 12h close > 1w EMA50, only short when 12h close < 1w EMA50.
- Volume confirmation: current 12h volume > 2.0 * 20-period 12h volume MA.
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian levels act as dynamic support/resistance.
- Exit: price reverts to midpoint of Donchian channel from prior 20 periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian levels from prior 20 periods (use completed 12h bar)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Mid = (Upper + Lower) / 2
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    mid = (upper + lower) / 2.0
    
    # Align Donchian levels to 12h timeframe (completed 20-bar lookback only)
    upper_aligned = align_htf_to_ltf(prices, prices, upper)
    lower_aligned = align_htf_to_ltf(prices, prices, lower)
    mid_aligned = align_htf_to_ltf(prices, prices, mid)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 12h close vs 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need 1w EMA50, Donchian lookback, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(mid_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper DONCH AND uptrend AND volume spike
            if close[i] > upper_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower DONCH AND downtrend AND volume spike
            elif close[i] < lower_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint or reverse signal
            if close[i] <= mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint or reverse signal
            if close[i] >= mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0