#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Uses daily Donchian channels (20-day high/low) for breakouts aligned with weekly trend
- 1w EMA50 defines longer-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.5x 20-day average) filters weak breakouts
- Designed for 1d timeframe targeting 10-25 trades/year (40-100 over 4 years)
- Works in both bull and bear markets by following the weekly trend direction
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
    
    # Calculate 1d Donchian channels (20-period)
    # Using rolling window on daily data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll  # 20-day high
    donchian_low = low_roll    # 20-day low
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # need Donchian(20), EMA50(1w), vol MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1w EMA50 AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1w EMA50 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level OR crosses 1w EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long when price < Donchian low OR < 1w EMA50
                if close[i] < donchian_low[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian high OR > 1w EMA50
                if close[i] > donchian_high[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0