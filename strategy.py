#!/usr/bin/env python3
"""
4H Donchian Breakout + 12H Trend + Volume Confirmation
Hypothesis: 4H Donchian(20) breakouts capture momentum; 12H EMA(21) filter aligns with higher timeframe trend; volume > 1.8x 30-period average confirms institutional participation. Designed for 4H to balance trade frequency and signal quality in bull/bear markets. Target: 25-45 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12H data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12H EMA(21) for trend filter
    ema_21_12h = df_12h['close'].ewm(span=21, adjust=False).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Donchian channels (20-period) on 4H
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.8x 30-period average on 4H)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < donchian_low[i] or close[i] < ema_21_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > donchian_high[i] or close[i] > ema_21_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long with trend alignment
            if (close[i] >= donchian_high[i] and 
                close[i] > ema_21_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short with trend alignment
            elif (close[i] <= donchian_low[i] and 
                  close[i] < ema_21_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals