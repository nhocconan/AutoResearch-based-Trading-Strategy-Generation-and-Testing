#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA Trend + Volume Confirmation
Hypothesis: 4h Donchian(20) breakouts aligned with 12h EMA(50) trend and volume spikes
capture strong momentum moves. Works in bull markets via breakouts and in bear markets
via breakdowns. Volume filter reduces false signals. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_ema_volume_v1"
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
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50 = df_12h['close'].ewm(span=50, adjust=False).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume filter (>1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA(20) or Donchian lower band
            ema_20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
            if close[i] < ema_20 or close[i] <= low_roll[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA(20) or Donchian upper band
            ema_20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
            if close[i] > ema_20 or close[i] >= high_roll[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long: price crosses above Donchian upper band with trend and volume
            if (close[i] > high_roll[i] and 
                close[i] > ema_50_4h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakdown short: price crosses below Donchian lower band with trend and volume
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_50_4h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals