#!/usr/bin/env python3
"""
1d Donchian Breakout + 1w EMA Trend + Volume Confirmation
Hypothesis: Donchian(20) breakouts on 1d capture major trend moves. 
1w EMA(20) filters for long-term trend alignment. Volume > 2x average 
confirms institutional participation. Works in bull/bear by only taking 
breakouts in direction of weekly trend, avoiding counter-trend whipsaws.
Target: 50-100 total trades over 4 years (12-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_ema_volume_v1"
timeframe = "1d"
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
    
    # Donchian channels (20-period) - use previous values to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter (>2x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    # 1w EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_20 = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low OR trend reverses
            if close[i] < low_20[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high OR trend reverses
            if close[i] > high_20[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above 20-day high with volume and trend alignment
            if (close[i] > high_20[i] and 
                close[i] > ema_20_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks below 20-day low with volume and trend alignment
            elif (close[i] < low_20[i] and 
                  close[i] < ema_20_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals