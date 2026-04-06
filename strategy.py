#!/usr/bin/env python3
"""
12h Donchian breakout with 1d EMA trend filter and volume confirmation.
Hypothesis: Donchian(20) breakouts in direction of daily trend with volume confirmation
capture strong momentum moves in both bull and bear markets while avoiding false breakouts.
Volume confirmation ensures breakouts have conviction. Works in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14296_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 12h timeframe (shifted by 1 day for completed bars only)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 50 for EMA)
    start = max(20, 20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to EMA or opposite Donchian band
        if position == 1:  # long position
            if close[i] <= ema_1d_aligned[i] or close[i] >= high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_1d_aligned[i] or close[i] <= low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend and volume confirmation
            # Long when price breaks above Donchian high in uptrend with volume
            # Short when price breaks below Donchian low in downtrend with volume
            long_breakout = close[i] > high_20_aligned[i]
            short_breakout = close[i] < low_20_aligned[i]
            uptrend = close[i] > ema_1d_aligned[i]
            downtrend = close[i] < ema_1d_aligned[i]
            
            if long_breakout and uptrend and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals