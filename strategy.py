#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Long: price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5x 20-period avg
- Short: price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5x 20-period avg
- Exit: price crosses 1d EMA34 OR Donchian breakout in opposite direction
- Uses discrete signal sizing (0.25) to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on primary timeframe (12h)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donch_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donch_low[i-1]  # Break below previous period's low
        
        if position == 0:
            # Long: Donchian breakout up + price > 1d EMA34 + volume confirmation
            if breakout_up and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + price < 1d EMA34 + volume confirmation
            elif breakout_down and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < 1d EMA34 OR Donchian breakout down
            if close[i] < ema_34_1d_aligned[i] or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > 1d EMA34 OR Donchian breakout up
            if close[i] > ema_34_1d_aligned[i] or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0