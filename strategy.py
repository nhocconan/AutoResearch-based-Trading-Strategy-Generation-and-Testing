# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
In both bull and bear markets, breakouts aligned with higher timeframe trend (12h EMA50) tend to continue.
Volume > 1.5x average confirms breakout strength. Target: 12-37 trades/year (50-150 over 4 years).
Position size: 0.25. Uses discrete levels to minimize fee churn.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    close_12h = df_12h['close'].values
    ema_50 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_50[i] = (close_12h[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 12h EMA50 to 6h timeframe (waits for 12h bar close)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 20-period Donchian channels on 6h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # 20-period average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 20 for Donchian, 20 for volume, 50 for EMA50 seed
    start_idx = max(lookback, vol_period, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 12h EMA50
        bullish = price > ema_50_aligned[i]
        bearish = price < ema_50_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long breakout: price breaks above Donchian high in bullish trend with volume
            if bullish and price > highest_high[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below Donchian low in bearish trend with volume
            elif bearish and price < lowest_low[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns bearish
            if price < lowest_low[i] or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns bullish
            if price > highest_high[i] or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0