#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA21 trend filter and volume spike confirmation.
- Donchian: upper = 20-period high, lower = 20-period low (using prior closed 4h bar for look-ahead safety)
- Long: Close > upper + volume > 2.0x 20-period avg + price > 12h EMA21
- Short: Close < lower + volume > 2.0x 20-period avg + price < 12h EMA21
- Exit: Opposite breakout (Close < upper for long, Close > lower for short) or 12h EMA21 trend flip
- Uses Donchian for structure, volume for conviction, 12h EMA21 for MTF trend filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- EMA21 provides adaptive trend filter to reduce false breakouts in choppy markets
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
    
    # Volume confirmation: > 2.0x 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA21 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate prior closed 4h Donchian levels (use shift(1) for look-ahead safety)
    # We need the high/low of the last 20 completed 4h bars (excluding current forming bar)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 21)  # Need 50 for EMA21 alignment, 21 for Donchian (20 + 1 for shift)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > upper + volume confirmation + price > 12h EMA21
            if (close[i] > donchian_upper[i] and 
                volume_confirm and 
                close[i] > ema_21_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < lower + volume confirmation + price < 12h EMA21
            elif (close[i] < donchian_lower[i] and 
                  volume_confirm and 
                  close[i] < ema_21_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < upper OR price < 12h EMA21 (trend flip)
            if close[i] < donchian_upper[i] or close[i] < ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > lower OR price > 12h EMA21 (trend flip)
            if close[i] > donchian_lower[i] or close[i] > ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0