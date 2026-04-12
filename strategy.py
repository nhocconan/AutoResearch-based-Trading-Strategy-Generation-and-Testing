#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_v2
Breakout strategy using Donchian channels on 4h with daily trend filter.
Enters on Donchian breakout with volume confirmation and daily trend alignment.
Exits on opposite Donchian breakout or trend reversal.
Designed for 20-40 trades/year to minimize fee drag.
Works in trending markets by following daily trend direction.
"""

name = "4h_1d_donchian_breakout_v2"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily 50 EMA for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian high with daily uptrend and volume
        if (close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below Donchian low with daily downtrend and volume
        elif (close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price breaks opposite Donchian level or trend reversal
        elif position == 1 and (close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]):
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