#!/usr/bin/env python3
"""
1h_4hDonchian20_1dEMA34_Trend
Strategy: 1h trend following using 4h Donchian(20) breakout + 1d EMA(34) filter.
Long: Price breaks above 4h Donchian upper + price > 1d EMA(34)
Short: Price breaks below 4h Donchian lower + price < 1d EMA(34)
Exit: Opposite Donchian break or EMA filter fails
Position size: 0.20
Uses 4h for trend direction/breakout, 1d for trend filter, 1h only for entry timing.
Session filter 08-20 UTC to reduce noise. Target 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Donchian breakout
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 1h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from EMA
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Breakout signals
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            # Long: Breakout above Donchian high + uptrend
            if breakout_up and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below Donchian low + downtrend
            elif breakout_down and downtrend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Breakdown below Donchian low OR trend fails
            if breakout_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Breakout above Donchian high OR trend fails
            if breakout_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_1dEMA34_Trend"
timeframe = "1h"
leverage = 1.0