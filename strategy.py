#!/usr/bin/env python3
"""
1d_Donchian20_1wEMA34_Trend_Volume
Hypothesis: Daily Donchian(20) breakouts filtered by weekly EMA34 trend and volume > 1.5x average.
Works in bull markets via breakout continuation and in bear via mean-reversion off extremes.
Target: 30-100 total trades over 4 years (~7-25/year) to avoid fee drag.
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
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on weekly close
    close_1w = df_1w['close'].values
    ema_period = 34
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA to daily timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Donchian channels (20-period)
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-20:i])
        lower_channel[i] = np.min(low[i-20:i])
    
    # Volume confirmation (20-period average)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Start after all indicators are ready
    start_idx = max(20, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below weekly EMA(34)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above upper channel in uptrend with volume
            if price > upper_channel[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower channel in downtrend with volume
            elif price < lower_channel[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below lower channel or trend reverses
            if price < lower_channel[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns above upper channel or trend reverses
            if price > upper_channel[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0