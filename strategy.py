#!/usr/bin/env python3
"""
12h Donchian Breakout + Daily Trend + Volume Confirmation
Hypothesis: Daily Donchian(20) breakouts aligned with daily EMA(50) trend and volume spikes
capture strong momentum moves. Works in bull markets via breakouts and in bear markets via
short breakdowns. Volume filter reduces false signals. Designed for 12h timeframe to limit
trade frequency and avoid fee drag. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Daily Donchian channels (20-period)
    donchian_high = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1)
    donchian_low = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1)
    
    # Daily EMA(50) for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean().shift(1)
    
    # Daily volume average (20-period)
    vol_ma = df_1d['volume'].rolling(window=20, min_periods=20).mean().shift(1)
    
    # Align to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high.values)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low.values)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50.values)
    vol_ma_12h = align_htf_to_ltf(prices, df_1d, vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < donchian_low_12h[i] or close[i] < ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > donchian_high_12h[i] or close[i] > ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian high with trend and volume
            if (close[i] >= donchian_high_12h[i] and 
                close[i] > ema_50_12h[i] and 
                volume[i] > vol_ma_12h[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Breakdown short below Donchian low with trend and volume
            elif (close[i] <= donchian_low_12h[i] and 
                  close[i] < ema_50_12h[i] and 
                  volume[i] > vol_ma_12h[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals