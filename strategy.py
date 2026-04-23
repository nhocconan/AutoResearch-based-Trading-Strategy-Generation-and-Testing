#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation
- Donchian(20) from 6h captures medium-term breakouts with controlled frequency
- 12h EMA34 defines the trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 1.8x 24-period MA) reduces false breakouts
- Designed for 6h timeframe to capture medium-term breakouts with target: 12-37 trades/year
- Uses Donchian breakouts which have proven effective on BTC/ETH in both bull and bear markets
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
    
    # Calculate 6h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 24)  # need Donchian, 12h EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 12h EMA34 AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below 12h EMA34 AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band OR crosses 12h EMA34
            exit_signal = False
            if position == 1:
                # Exit long when price < Donchian lower OR < 12h EMA34
                if close[i] < low_ma[i] or close[i] < ema_34_12h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian upper OR > 12h EMA34
                if close[i] > high_ma[i] or close[i] > ema_34_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hEMA34_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0