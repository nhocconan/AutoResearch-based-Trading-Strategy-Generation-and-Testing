#!/usr/bin/env python3
"""
12h Donchian Breakout with Weekly EMA Trend and Volume Confirmation
Hypothesis: 12h timeframe reduces trade frequency to avoid fee drag while capturing major trends.
Donchian(20) breakouts aligned with weekly EMA(50) trend and volume confirmation (>1.5x average)
provide high-quality entries. Works in bull markets (breakout momentum) and bear markets
(mean reversion at opposite band with trend filter). Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_weekly_ema_volume_v1"
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
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend reverses
            if close[i] <= low_roll[i] or close[i] < ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend reverses
            if close[i] >= high_roll[i] or close[i] > ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at upper band with trend alignment
            if (close[i] >= high_roll[i] and 
                close[i] > ema_50_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at lower band with trend alignment
            elif (close[i] <= low_roll[i] and 
                  close[i] < ema_50_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Mean reversion long at lower band with trend alignment (bear market bounce)
            elif (close[i] <= low_roll[i] and 
                  close[i] > ema_50_12h[i] and 
                  vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short at upper band with trend alignment (bull market pullback)
            elif (close[i] >= high_roll[i] and 
                  close[i] < ema_50_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals