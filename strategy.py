#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Uses 4h timeframe (primary) and 12h HTF for EMA50 trend alignment
- Donchian levels calculated from 20-period high/low: upper = max(high,20), lower = min(low,20)
- Breakout logic: long when price crosses above upper band with volume confirmation and uptrend, short when price crosses below lower band with volume confirmation and downtrend
- Trend filter: only long when price > 12h EMA50, only short when price < 12h EMA50
- Volume confirmation: current volume > 1.8 * 30-period volume MA to avoid low-volume false signals
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Exit: reverse signal or price reverts to 20-period EMA (mean reversion)
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 30-period volume MA
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * volume_ma)
    
    # Trend filter: price above/below 12h EMA50
    uptrend = close > ema_50_12h_aligned
    downtrend = close < ema_50_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 20)  # Need 12h EMA50, volume MA(30), Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper Donchian band AND uptrend AND volume confirmation
            if close[i] > high_ma[i] and close[i-1] <= high_ma[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower Donchian band AND downtrend AND volume confirmation
            elif close[i] < low_ma[i] and close[i-1] >= low_ma[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to 20-period EMA (mean reversion) or reverse signal
            ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
            if close[i] <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to 20-period EMA (mean reversion) or reverse signal
            ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
            if close[i] >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0