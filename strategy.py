#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
- Uses 6h Donchian channel (20-bar) for breakout detection
- Weekly EMA50 defines higher timeframe trend: only trade breakouts in trend direction
- Volume confirmation (> 2.0x 20-period average) filters false breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Weekly trend filter reduces whipsaws in sideways markets
- Works in both bull and bear markets by trading with the weekly trend
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
    
    # Calculate 6h Donchian channel (20-bar)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper with weekly uptrend and volume
            long_breakout = (close[i] > high_20[i] and 
                           close[i] > ema_50_1w_aligned[i] and
                           volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian lower with weekly downtrend and volume
            short_breakout = (close[i] < low_20[i] and 
                            close[i] < ema_50_1w_aligned[i] and
                            volume[i] > 2.0 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower or weekly trend turns bearish
                if (close[i] < low_20[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian upper or weekly trend turns bullish
                if (close[i] > high_20[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0