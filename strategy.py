#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Long: Close > Donchian Upper(20) AND price > 1w EMA50 AND volume > 2.0x 20-period avg
- Short: Close < Donchian Lower(20) AND price < 1w EMA50 AND volume > 2.0x 20-period avg
- Exit: Opposite Donchian breakout OR price crosses 1w EMA50
- Uses 1w HTF for EMA50 trend filter and Donchian from 6h timeframe (self-contained)
- Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe
- Donchian channels provide structure in trending markets, 1w EMA50 filters major trend
- Volume confirmation reduces false breakouts in choppy conditions
- Works in both bull (trend following) and bear (short opportunities) via EMA50 filter
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 6h timeframe (self-contained)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for 1w EMA, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs current channels)
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        if position == 0:
            # Long: Donchian upper breakout AND price > 1w EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout AND price < 1w EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakout OR price < 1w EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper breakout OR price > 1w EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0