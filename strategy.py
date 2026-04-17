#!/usr/bin/env python3
"""
4h_Donchian20_1dEMA21_VolumeFilter_V1
Strategy: 4h Donchian(20) breakout with 1d EMA21 trend filter and volume confirmation.
Long: Price breaks above 20-period high + price > 1d EMA21 + volume > 1.5x 20-period avg
Short: Price breaks below 20-period low + price < 1d EMA21 + volume > 1.5x 20-period avg
Exit: Opposite Donchian breakout or trend reversal
Position size: 0.25
Designed to work in both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA21
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate daily EMA21
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = high
    low_4h = low
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume average (20-period)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(20, n):  # warmup for Donchian
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout signals
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Trend filter
        uptrend = close[i] > ema_21_1d_aligned[i]
        downtrend = close[i] < ema_21_1d_aligned[i]
        
        if position == 0:
            # Long: Breakout above Donchian high + uptrend + volume
            if breakout_up and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low + downtrend + volume
            elif breakout_down and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Breakdown below Donchian low or trend reversal
            if breakout_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Breakout above Donchian high or trend reversal
            if breakout_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA21_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0