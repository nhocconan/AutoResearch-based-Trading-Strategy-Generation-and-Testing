#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
Works in bull/bear: breakouts capture momentum, volume confirms legitimacy, trend filter avoids counter-trend whipsaws.
Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) - 20-period high/low
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 20 for volume avg, 50 for 1d EMA
    start_idx = max(lookback, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Conservative position size to manage drawdown
        
        if position == 0:
            # Flat - look for breakout with volume and trend confirmation
            # Long: break above Donchian high + volume confirm + 1d EMA50 rising
            long_entry = (close_val > donchian_high[i]) and \
                       volume_confirm[i] and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1])
            # Short: break below Donchian low + volume confirm + 1d EMA50 falling
            short_entry = (close_val < donchian_low[i]) and \
                        volume_confirm[i] and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1])
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price breaks below Donchian low (contrarian)
            if close_val < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian high (contrarian)
            if close_val > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0