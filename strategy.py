#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper AND 1w close > EMA50 AND volume > 1.5x 20-bar avg.
Short when price breaks below Donchian lower AND 1w close < EMA50 AND volume > 1.5x 20-bar avg.
Exit when price touches Donchian midpoint or opposite band.
Uses 1w for trend filter, 12h for execution and volume confirmation.
Designed to capture medium-term trends with volatility-based entries and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Donchian(20) channels
    # Upper = max(high, 20), Lower = min(low, 20)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate 12h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > donchian_upper[i]
        breakout_lower = close[i] < donchian_lower[i]
        
        # Exit conditions: touch midpoint or opposite band
        touch_mid = abs(close[i] - donchian_mid[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < donchian_lower[i]) or \
                         (position == -1 and close[i] > donchian_upper[i])
        
        if position == 0:
            # Long: break above upper with uptrend and volume confirmation
            if (breakout_upper and uptrend and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: break below lower with downtrend and volume confirmation
            elif (breakout_lower and downtrend and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint or break below lower
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint or break above upper
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Volume_Trend"
timeframe = "12h"
leverage = 1.0