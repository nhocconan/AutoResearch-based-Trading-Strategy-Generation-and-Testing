#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Long: Close breaks above Donchian upper(20) + price > 12h EMA50 + volume > 1.8x 20-period avg
- Short: Close breaks below Donchian lower(20) + price < 12h EMA50 + volume > 1.8x 20-period avg
- Exit: Close crosses Donchian midpoint (mean reversion to median)
- Uses Donchian channel for structure, 12h EMA for trend filter, volume for breakout strength
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
- Works in bull markets (breakouts with trend alignment) and bear markets (mean reversion at extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donch_hi = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_hi + donch_lo) / 2
    
    # Volume confirmation: > 1.8x 20-period average (tighter filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_hi[i]) or 
            np.isnan(donch_lo[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average - tighter)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian HI + above 12h EMA50 + volume confirmation
            if (close[i] > donch_hi[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian LO + below 12h EMA50 + volume confirmation
            elif (close[i] < donch_lo[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Donchian midpoint (mean reversion)
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Donchian midpoint (mean reversion)
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0