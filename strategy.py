#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_trend
4h Donchian breakout with 1d trend filter (EMA50) and volume confirmation.
Long when price breaks above Donchian(20) + price > EMA50(1d) + volume spike.
Short when price breaks below Donchian(20) + price < EMA50(1d) + volume spike.
Exit when price crosses back below/above Donchian middle.
Target: 25-40 trades/year to minimize fee drag. Works in bull/bear via trend filter.
"""

name = "4h_1d_donchian_breakout_volume_trend"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_roll + low_roll) / 2.0
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: break above upper band + uptrend + volume
        if (close[i] > high_roll[i] and close[i] > ema_50_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: break below lower band + downtrend + volume
        elif (close[i] < low_roll[i] and close[i] < ema_50_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back below/above middle
        elif position == 1 and close[i] < donchian_middle[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_middle[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals