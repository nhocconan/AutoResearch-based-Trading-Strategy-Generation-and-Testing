#!/usr/bin/env python3
# 12h_1d_Donchian_Breakout_Trend_Filter
# Hypothesis: Uses 1d Donchian channels to establish trend direction and 12h Donchian breakouts for entry timing.
# In uptrend (price above 1d upper band), go long on 12h breakout above upper band; in downtrend (price below 1d lower band), go short on 12h breakout below lower band.
# Volume confirmation (>1.5x 20-period average) filters for institutional participation.
# Designed for low trade frequency (<200 total 12h trades) to minimize fee drag.
# Works in bull/bear markets by following 1d trend while using 12h breaks for precise entries.

name = "12h_1d_Donchian_Breakout_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Donchian trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian channels (20-period)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Determine trend: above upper band = uptrend, below lower band = downtrend
    trend_long = close_1d > donch_high_1d
    trend_short = close_1d < donch_low_1d
    
    # Align 1d trend to 12h timeframe
    trend_long_aligned = align_htf_to_ltf(prices, df_1d, trend_long)
    trend_short_aligned = align_htf_to_ltf(prices, df_1d, trend_short)
    
    # 12h Donchian channels for entry/exit (20-period)
    donch_high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(trend_long_aligned[i]) or
            np.isnan(trend_short_aligned[i]) or
            np.isnan(donch_high_12h[i]) or
            np.isnan(donch_low_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend on 1d + price breaks above 12h upper band + volume spike
            if (trend_long_aligned[i] and 
                close[i] > donch_high_12h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend on 1d + price breaks below 12h lower band + volume spike
            elif (trend_short_aligned[i] and 
                  close[i] < donch_low_12h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 12h lower band OR 1d trend turns down
            if (close[i] < donch_low_12h[i]) or \
               not trend_long_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 12h upper band OR 1d trend turns up
            if (close[i] > donch_high_12h[i]) or \
               not trend_short_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals