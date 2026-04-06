#!/usr/bin/env python3
"""
4h Donchian breakout with 12h EMA trend and volume confirmation.
Hypothesis: Breakouts of Donchian(20) channels aligned with 12h trend capture strong moves in both bull and bear markets, while volume confirmation filters false breakouts. Works in all regimes due to trend filter and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14293_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_12h = calculate_ema(close_12h, 50)
    
    # Align to 4h timeframe (shifted by 1 bar for completed bars only)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
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
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 50 for EMA)
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to opposite Donchian band or EMA crossover
        if position == 1:  # long position
            if close[i] <= low_20[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= high_20[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout of Donchian channel with trend and volume confirmation
            # Long when price breaks above upper band in uptrend with volume
            # Short when price breaks below lower band in downtrend with volume
            long_breakout = close[i] > high_20[i]
            short_breakout = close[i] < low_20[i]
            
            long_setup = long_breakout and (close[i] > ema_12h_aligned[i]) and vol_confirm[i]
            short_setup = short_breakout and (close[i] < ema_12h_aligned[i]) and vol_confirm[i]
            
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