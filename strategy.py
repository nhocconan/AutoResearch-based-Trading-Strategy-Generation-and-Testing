#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
- Uses 12h timeframe to reduce trade frequency and fee drag
- Donchian(20) breakout provides clear entry signals in trending markets
- 1d EMA50 defines primary trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.5x 20-period average) filters weak breakouts
- Designed for 12h timeframe targeting 12-30 trades/year (50-120 over 4 years)
- Works in both bull and bear markets by following the 1d EMA50 trend filter
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # need 1d EMA50, Donchian(20), vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above 1d EMA50 AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND below 1d EMA50 AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level OR crosses 1d EMA50
            exit_signal = False
            if position == 1:
                # Exit long when price < lower Donchian OR < 1d EMA50
                if close[i] < low_ma[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > upper Donchian OR > 1d EMA50
                if close[i] > high_ma[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0