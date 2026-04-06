#!/usr/bin/env python3
"""
4h Donchian breakout with 1d trend filter and volume confirmation.
Hypothesis: Donchian(20) breakouts in the direction of the 1-day EMA trend,
filtered by volume spikes, capture strong momentum moves while avoiding chop.
Works in both bull (breakouts continue) and bear (breakdowns continue) markets.
Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14289_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA)
    start = max(20, 50)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or volatility stop
        if position == 1:  # long position
            if low[i] <= low_roll[i]:  # Donchian lower break = exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if high[i] >= high_roll[i]:  # Donchian upper break = exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend and volume confirmation
            # Long when price breaks above Donchian upper band in uptrend
            # Short when price breaks below Donchian lower band in downtrend
            long_setup = (high[i] >= high_roll[i]) and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
            short_setup = (low[i] <= low_roll[i]) and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals