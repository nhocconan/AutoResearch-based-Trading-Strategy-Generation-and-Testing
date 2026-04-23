#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Long: Close > Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x 20-period avg
- Short: Close < Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x 20-period avg
- Exit: Opposite Donchian breakout OR price crosses 12h EMA50
- Works in bull (buy breakouts) and bear (sell breakdowns)
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
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
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for safety, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donch_high[i-1]  # Close above previous period's high
        breakout_down = close[i] < donch_low[i-1]  # Close below previous period's low
        
        if position == 0:
            # Long: Donchian breakout up AND price > 12h EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND price < 12h EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakout down OR price < 12h EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up OR price > 12h EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0