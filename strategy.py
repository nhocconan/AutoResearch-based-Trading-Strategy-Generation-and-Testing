#!/usr/bin/env python3

"""
Hypothesis: 6-hour Price Channel Breakout with 1-week Momentum Filter and Volume Spike.
Uses Donchian channels (20-period) on 6h for breakout signals, filtered by 1-week RSI momentum
to avoid counter-trend trades, and confirmed by volume spikes. The weekly RSI acts as a
strong trend filter that works in both bull and bear markets by identifying momentum
exhaustion and continuation. Target: 15-35 trades/year per symbol (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel: upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian channel - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian Channel (20-period)
    donchian_upper, donchian_lower = calculate_donchian(
        df_6h['high'].values, df_6h['low'].values, 20
    )
    
    # Align Donchian bands to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # Load 1w data for momentum filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week RSI (14-period) for momentum filter
    rsi_1w = calculate_rsi(df_1w['close'].values, 14)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
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
        
        if position == 0:
            # Long: price breaks above upper Donchian, weekly RSI > 50 (bullish momentum), volume spike
            if (close[i] > donchian_upper_aligned[i] and
                rsi_1w_aligned[i] > 50 and
                volume[i] > 2.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, weekly RSI < 50 (bearish momentum), volume spike
            elif (close[i] < donchian_lower_aligned[i] and
                  rsi_1w_aligned[i] < 50 and
                  volume[i] > 2.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of channel or opposite breakout
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian or returns to middle
                middle = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2
                if close[i] < donchian_lower_aligned[i] or close[i] < middle:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian or returns to middle
                middle = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2
                if close[i] > donchian_upper_aligned[i] or close[i] > middle:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_1wRSI_Momentum_Volume"
timeframe = "6h"
leverage = 1.0