#!/usr/bin/env python3
"""
4H Donchian Breakout + Daily EMA + Volume Confirmation
Hypothesis: 4-hour Donchian(20) breakouts in the direction of the daily EMA(50) trend,
confirmed by volume (>1.5x 20-period average), capture strong momentum moves.
Exit when price crosses the opposite Donchian band or daily EMA. Designed for 4h
to balance trade frequency and signal strength in both bull and bear markets.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_daily_ema_volume_v1"
timeframe = "4h"
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
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or trend reverses
            if close[i] <= donchian_low[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or trend reverses
            if close[i] >= donchian_high[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian high with trend alignment
            if (close[i] >= donchian_high[i] and 
                close[i] > ema_50_4h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short below Donchian low with trend alignment
            elif (close[i] <= donchian_low[i] and 
                  close[i] < ema_50_4h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals