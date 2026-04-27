#!/usr/bin/env python3
"""
6h_LongTermTrend_VolumeBreakout
Hypothesis: Use 12h EMA trend as primary filter, enter on 6h Donchian(20) breakout with volume spike confirmation.
Works in bull via breakouts above rising 12h EMA, bear via breakouts below falling 12h EMA.
Volume filter ensures breakouts have conviction. Designed for 50-150 trades over 4 years.
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
    
    # Get 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 6h
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume spike filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(50, donchian_period, 20)
    
    for i in range(start_idx, n):
        # Skip if EMA not ready
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_12h_aligned[i]
        donchian_upper = upper[i]
        donchian_lower = lower[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 12h EMA AND volume spike
            if close[i] > donchian_upper and close[i] > ema_trend and vol_ok:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower AND below 12h EMA AND volume spike
            elif close[i] < donchian_lower and close[i] < ema_trend and vol_ok:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR drops below 12h EMA
            if close[i] < donchian_lower or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR rises above 12h EMA
            if close[i] > donchian_upper or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_LongTermTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0