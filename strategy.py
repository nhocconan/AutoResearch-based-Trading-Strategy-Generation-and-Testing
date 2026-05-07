# !/usr/bin/env python3
# 12h_Donchian20_1dTrend_VolumeSpike
# Hypothesis: 12h Donchian(20) breakout with 1-day EMA34 trend filter and volume spike confirmation.
# In bull markets: price above EMA34 + breaks upper Donchian(20) with volume = long.
# In bear markets: price below EMA34 + breaks lower Donchian(20) with volume = short.
# Daily EMA34 filter reduces whipsaws; volume spike confirms breakout strength.
# Target: 20-30 trades/year to minimize fee drag while maintaining edge.

name = "12h_Donchian20_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian(20) on 12h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter on 12h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_34_1d_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > upper Donchian, above 1d EMA34 trend, volume spike
            if close[i] > high_max_20[i] and close[i] > ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < lower Donchian, below 1d EMA34 trend, volume spike
            elif close[i] < low_min_20[i] and close[i] < ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < upper Donchian or below 1d EMA34 trend
            if close[i] < high_max_20[i] or close[i] < ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > lower Donchian or above 1d EMA34 trend
            if close[i] > low_min_20[i] or close[i] > ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals