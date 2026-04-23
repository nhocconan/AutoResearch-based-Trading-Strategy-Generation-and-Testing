#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Donchian(20): 20-period high/low on 1d timeframe for breakout signals
- Long: Price breaks above 20d high + volume > 2x 20d avg volume + price > 1w EMA50 (uptrend)
- Short: Price breaks below 20d low + volume > 2x 20d avg volume + price < 1w EMA50 (downtrend)
- Exit: Opposite Donchian breakout (short break for long exit, long break for short exit)
- Uses Donchian for structure, volume for conviction, 1w EMA50 for HTF trend filter
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Donchian breakouts work in both bull (buy breakouts) and bear (sell breakdowns) markets
"""

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
    
    # Volume confirmation: > 2x 20-period average (strict to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Donchian(20) for breakout signals
    highest_high_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_high_20d[i]) or
            np.isnan(lowest_low_20d[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20d[i-1]  # Break above prior 20d high
        breakout_down = close[i] < lowest_low_20d[i-1]   # Break below prior 20d low
        
        if position == 0:
            # Long: Donchian breakout up + volume confirmation + price > 1w EMA50 (uptrend)
            if breakout_up and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume confirmation + price < 1w EMA50 (downtrend)
            elif breakout_down and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakout down (opposite breakout)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up (opposite breakout)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0