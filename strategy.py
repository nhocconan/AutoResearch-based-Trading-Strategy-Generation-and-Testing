#!/usr/bin/env python3

"""
Hypothesis: 1-day Donchian channel breakout with 1-week EMA trend filter and volume confirmation.
Breakouts above the 20-day high or below the 20-day low are taken only when aligned with the weekly trend.
Volume spikes confirm institutional participation. This strategy captures medium-term trends while avoiding whipsaw
by requiring weekly trend alignment. Works in both bull and bear markets by trading with the higher timeframe trend.
Target: 7-25 trades/year per symbol (30-100 total over 4 years).
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
    
    # Load 1d data for Donchian channel - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian Channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian Channel components
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA for trend filter (50-period)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-calculate average volume for volume confirmation (20-period)
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_vol_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        if position == 0 and vol_confirmed:
            # Long: price breaks above Donchian high, above 1w EMA (uptrend)
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below 1w EMA (downtrend)
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band or trend changes
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Donchian low or closes below 1w EMA
                if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches Donchian high or closes above 1w EMA
                if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0