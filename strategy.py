#!/usr/bin/env python3
"""
1d Donchian(20) breakout with weekly trend filter and volume confirmation.
Hypothesis: Breakouts above/below 20-day high/low aligned with weekly trend capture strong moves, while volume confirmation filters false signals. Works in both bull and bear markets by following trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14298_1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema_1w = calculate_ema(close_1w, 20)
    
    # Align to daily timeframe (shifted by 1 week for completed bars only)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily data
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
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Check exits: price crosses back through the 20-day mid-point or opposite channel
        mid_point = (high_20[i] + low_20[i]) / 2
        
        if position == 1:  # long position
            if close[i] <= mid_point or close[i] <= low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= mid_point or close[i] >= high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend and volume confirmation
            # Long when price breaks above 20-day high in uptrend
            # Short when price breaks below 20-day low in downtrend
            long_setup = (close[i] > high_20[i]) and (close[i] > ema_1w_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] < low_20[i]) and (close[i] < ema_1w_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals