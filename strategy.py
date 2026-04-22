#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian Channel breakout with 12-hour EMA trend filter and volume confirmation.
The Donchian Channel (20-period high/low) identifies breakouts in both directions.
The 12-hour EMA (50-period) filters trades to follow the higher timeframe trend.
Volume spikes confirm institutional participation at breakout points.
This strategy aims to capture strong momentum moves with minimal trades to reduce fee drag.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA (50-period) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian Channel (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_avg_20[i])):
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
            # Long: price breaks above Donchian high, above 12h EMA, volume spike
            if (close[i] > donchian_high[i] and                    # Break above Donchian high
                close[i] > ema_50_12h_aligned[i] and               # Above 12h EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_20[i]):                  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below 12h EMA, volume spike
            elif (close[i] < donchian_low[i] and                   # Break below Donchian low
                  close[i] < ema_50_12h_aligned[i] and             # Below 12h EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_20[i]):                # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or crosses 12h EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low or below 12h EMA
                if close[i] < donchian_low[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high or above 12h EMA
                if close[i] > donchian_high[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0