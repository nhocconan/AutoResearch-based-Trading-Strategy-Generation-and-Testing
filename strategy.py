#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1w EMA(20) trend filter and volume confirmation.
Hypothesis: Donchian breakouts aligned with weekly trend capture strong momentum moves,
while volume confirmation filters false breakouts. Weekly EMA ensures we trade with the
higher timeframe trend, improving performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14288_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(20) for trend filter
    ema_1w = calculate_ema(close_1w, 20)
    
    # Align to 12h timeframe (shifted by 1 week for completed bars only)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to opposite Donchian level
        if position == 1:  # long position
            if close[i] <= low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Donchian breakouts with trend and volume confirmation
            # Long when price breaks above upper Donchian with uptrend and volume
            # Short when price breaks below lower Donchian with downtrend and volume
            long_setup = (close[i] > high_20[i]) and (close[i] > ema_1w_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < low_20[i]) and (close[i] < ema_1w_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals