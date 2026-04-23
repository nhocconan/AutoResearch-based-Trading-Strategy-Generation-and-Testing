#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout with 12h EMA34 Trend Filter and Volume Spike
- Uses Donchian channel breakout (20-period) for entry signals
- 12h EMA34 defines higher timeframe trend: only trade in direction of 12h trend
- Volume confirmation (> 2.0x 20-period average) filters weak breakouts
- Exit when price returns to Donchian midpoint or trend reverses
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in both bull and bear markets by trading breakouts with trend filter
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
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # for EMA34, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper AND above 12h EMA34 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND below 12h EMA34 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Donchian midpoint OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price closes below Donchian midpoint OR below 12h EMA34
                if (close[i] < donchian_mid[i] or close[i] < ema_34_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price closes above Donchian midpoint OR above 12h EMA34
                if (close[i] > donchian_mid[i] or close[i] > ema_34_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0