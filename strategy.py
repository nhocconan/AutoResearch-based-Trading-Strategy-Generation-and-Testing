#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation.
- Uses 1w timeframe for HTF trend alignment (more stable for 6h entries)
- Donchian breakout from previous 6h bar: upper = max(high[-20:]), lower = min(low[-20:])
- Long when price breaks above upper AND price > 1w EMA50 (uptrend) AND volume > 1.5 * volume MA(20)
- Short when price breaks below lower AND price < 1w EMA50 (downtrend) AND volume > 1.5 * volume MA(20)
- Exit when price reverts to mid-band of Donchian channel
- Discrete signal size: 0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) - within 6h limits
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w OHLC for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Calculate Donchian channels from previous completed 6h bar
    # We need to calculate this on 6h data, but we don't have direct access
    # Instead, we'll use the current prices to approximate with lookback
    # Using rolling window on the primary timeframe (6h)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1w EMA50, Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend AND volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to mid-band
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to mid-band
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0