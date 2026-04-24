#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Donchian channel breakouts capture momentum with clear entry/exit levels.
- 12h EMA50 provides higher-timeframe trend filter to reduce counter-trend trades.
- Volume spike (>2.0x 20-period average) confirms breakout validity.
- Discrete position sizing (0.30) balances return potential with fee drag control.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe.
- Works in bull/bear markets via 12h trend filter and volatility-based volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channel (20-period) on primary timeframe
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and above 12h EMA50 (bullish higher-timeframe trend)
            if close[i] > high_max[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below lower Donchian with volume spike and below 12h EMA50 (bearish higher-timeframe trend)
            elif close[i] < low_min[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian OR below 12h EMA50 (trend change)
            if close[i] < low_min[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price closes above upper Donchian OR above 12h EMA50 (trend change)
            if close[i] > high_max[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0