#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation.
Hypothesis: Price breaking 20-period Donchian channel on 4h, aligned with daily trend, captures breakouts with momentum while volume filters false signals. Works in both bull (breakouts up) and bear (breakouts down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14300_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to middle of Donchian channel or opposite breakout
        if position == 1:  # long position
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Long when price breaks above Donchian high in uptrend
            # Short when price breaks below Donchian low in downtrend
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            # Trend filter: price above/below 1d EMA
            uptrend = close[i] > ema_1d_aligned[i]
            downtrend = close[i] < ema_1d_aligned[i]
            
            if long_breakout and uptrend and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and downtrend and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals