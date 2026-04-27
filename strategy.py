#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 34-period Donchian breakout with 1-day EMA50 trend filter.
Breakouts occur when price breaks above/below 34-bar Donchian channels, filtered by
daily EMA50 direction to ensure alignment with higher timeframe trend. Volume > 1.5x
average confirms breakout strength. Uses discrete position sizes (±0.25) to minimize
fee churn. Target: 15-30 trades/year (60-120 total over 4 years). Works in bull/bear
by capturing breakouts aligned with daily trend.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA50 to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (34-period) on 6h data
    donchian_period = 34
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need EMA (49), Donchian (33), volume MA (19)
    start_idx = max(49, donchian_period-1, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = price > ema_1d_aligned[i]
        price_below_ema = price < ema_1d_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above upper channel in uptrend with volume
            if price_above_ema and price > upper_channel[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower channel in downtrend with volume
            elif price_below_ema and price < lower_channel[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower channel or trend reverses
            if price < lower_channel[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper channel or trend reverses
            if price > upper_channel[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian34_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0