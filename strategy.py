#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Long: Price breaks above Donchian(20) high + price > 12h EMA50 + volume > 2.0x 20-period avg
- Short: Price breaks below Donchian(20) low + price < 12h EMA50 + volume > 2.0x 20-period avg
- Exit: Opposite Donchian breakout or price crosses 12h EMA50
- Uses Donchian for structure, 12h EMA50 for HTF trend, volume spike for conviction
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)
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
    
    # Volume confirmation: > 2.0x 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average - tighter threshold)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donch_low[i-1]  # Break below previous period's low
        
        if position == 0:
            # Long: Donchian breakout up + price > 12h EMA50 + volume confirmation
            if (breakout_up and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + price < 12h EMA50 + volume confirmation
            elif (breakout_down and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakout down OR price < 12h EMA50
            if breakout_down or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up OR price > 12h EMA50
            if breakout_up or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0