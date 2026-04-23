#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout with 1w EMA50 trend filter and volume confirmation.
- Donchian(20): Upper/lower bands from 20-period high/low - price channel structure
- 1w EMA50: Weekly trend filter - ensures trades align with higher timeframe momentum
- Volume confirmation: > 1.8x 20-period average volume - filters low-quality breakouts
- Long: Close > Donchian Upper + price > 1w EMA50 + volume confirmation
- Short: Close < Donchian Lower + price < 1w EMA50 + volume confirmation
- Exit: Opposite Donchian breakout or price crosses 1w EMA50
- Uses Donchian for breakout structure, 1w EMA50 for HTF trend alignment, volume for confirmation
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts above EMA50) and bear markets (breakouts below EMA50)
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian Channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Donchian Upper + price > 1w EMA50 + volume confirmation
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian Lower + price < 1w EMA50 + volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Donchian Lower OR price < 1w EMA50
            if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Donchian Upper OR price > 1w EMA50
            if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0