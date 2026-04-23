#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation.
- Long: Close > Donchian High(20) AND price > 12h EMA50 (uptrend) AND volume > 2.0x 20-period avg
- Short: Close < Donchian Low(20) AND price < 12h EMA50 (downtrend) AND volume > 2.0x 20-period avg
- Exit: Opposite Donchian breakout OR price crosses 12h EMA50 (trend change)
- Volume spike reduces false breakouts in low-participation markets
- Donchian provides clear structure, EMA50 filters counter-trend trades
- Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag on 4h timeframe
- Works in bull (breakouts with trend) and bear (breakouts with trend continuation)
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
    
    # Volume confirmation: > 2.0x 20-period average (strong spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 for 12h trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Donchian High(20) AND price > 12h EMA50 (uptrend) AND volume spike
            if volume_spike and close[i] > donch_high[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian Low(20) AND price < 12h EMA50 (downtrend) AND volume spike
            elif volume_spike and close[i] < donch_low[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Donchian Low(20) OR price < 12h EMA50 (trend break)
            if close[i] < donch_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Donchian High(20) OR price > 12h EMA50 (trend break)
            if close[i] > donch_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0