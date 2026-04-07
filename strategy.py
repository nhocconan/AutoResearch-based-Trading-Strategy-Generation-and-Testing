#!/usr/bin/env python3
"""
4h_donchian_breakout_12h_trend_volume_v1
Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation. 
In trending markets, price breaks out of Donchian channels and continues in the direction of the trend.
Volume confirms institutional participation. Works in both bull and bear markets by following 12h trend.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # 4h Donchian channel (20-period)
    donchian_len = 20
    high_max = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    low_min = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_20 = df_12h['close'].ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend turns bearish
            if close[i] <= low_min[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend turns bullish
            if close[i] >= high_max[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume and bullish trend
            if (close[i] > high_max[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and bearish trend
            elif (close[i] < low_min[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals