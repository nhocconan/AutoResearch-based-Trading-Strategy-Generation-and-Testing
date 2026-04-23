#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA(50) trend filter + volume confirmation (>1.5x 20-period average)
- Donchian breakout captures momentum in trending markets
- 12h EMA(50) ensures trades align with higher-timeframe trend to avoid counter-trend whipsaws
- Volume confirmation (>1.5x average) validates breakout strength
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by trading with 12h trend from daily breakouts
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels for 4h timeframe
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (no extra delay needed as it's based on completed 4h bar)
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low)
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA, Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        # Long: price breaks above upper Donchian channel
        # Short: price breaks below lower Donchian channel
        long_breakout = close[i] > highest_high_aligned[i]
        short_breakout = close[i] < lowest_low_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long conditions: breakout above upper channel, uptrend, volume spike
            long_signal = (long_breakout and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: breakout below lower channel, downtrend, volume spike
            short_signal = (short_breakout and 
                           downtrend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian channel or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower Donchian channel or trend turns down
                if (close[i] < lowest_low_aligned[i] or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper Donchian channel or trend turns up
                if (close[i] > highest_high_aligned[i] or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0