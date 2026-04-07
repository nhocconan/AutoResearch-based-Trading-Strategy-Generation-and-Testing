#!/usr/bin/env python3
"""
6h Donchian Breakout + 12h Trend Filter + Volume Confirmation
Hypothesis: Donchian(20) breakouts on 6h timeframe capture momentum moves.
Trend filtered by 12h EMA(50) ensures alignment with higher timeframe trend.
Volume > 1.5x average confirms institutional participation. Works in both bull
and bear markets by filtering breakouts with trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_12h_trend_volume_v1"
timeframe = "6h"
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
    
    # 12h data for Donchian and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Donchian channels (20-period) on 12h
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend filter
    ema_50 = df_12h['close'].ewm(span=50, adjust=False).min_periods(50).mean().values
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    donchian_high_6h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume filter (>1.5x 24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < donchian_low_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > donchian_high_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at Donchian high with uptrend and volume
            if (close[i] >= donchian_high_6h[i] and 
                close[i] > ema_50_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at Donchian low with downtrend and volume
            elif (close[i] <= donchian_low_6h[i] and 
                  close[i] < ema_50_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals