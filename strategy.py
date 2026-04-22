#!/usr/bin/env python3

"""
Hypothesis: 6-hour Donchian breakout with weekly trend filter and volume confirmation.
Trades breakouts of 6-hour Donchian channels only when aligned with weekly trend direction.
Uses volume spike to confirm institutional participation. Designed for low trade frequency
(12-37 trades/year) to minimize fee drift and work in both bull and bear markets by
filtering with weekly trend and requiring volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (50-period)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6-hour Donchian channel (20-period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above Donchian upper with weekly uptrend
            if close[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with weekly downtrend
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower or weekly trend turns down
                if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper or weekly trend turns up
                if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_Breakout_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0