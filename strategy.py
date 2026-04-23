#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d, HTF: 1w for trend filter
- Long: Close breaks above 20-day high + price > 1w EMA50 (uptrend) + volume > 2.0x 20-period avg
- Short: Close breaks below 20-day low + price < 1w EMA50 (downtrend) + volume > 2.0x 20-period avg
- Exit: Close reverts to 10-day EMA (mean reversion to intermediate trend)
- Uses tighter volume confirmation (2.0x) to reduce trade frequency and fee drag
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires weekly trend alignment to avoid SOL-only bias
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
- Uses mtf_data helper for proper HTF alignment without look-ahead
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
    
    # Volume confirmation: > 2.0x 20-period average (tight volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d 10-period EMA for exit (mean reversion target)
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema_10[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above 20-day high + price > 1w EMA50 (uptrend) + volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below 20-day low + price < 1w EMA50 (downtrend) + volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to 10-day EMA (mean reversion)
            if close[i] <= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to 10-day EMA (mean reversion)
            if close[i] >= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0