#!/usr/bin/env python3
# 1D_1W_Donchian_Breakout_Trend_Filter
# Hypothesis: Uses weekly trend (Donchian(10) direction) to filter daily Donchian(20) breakouts.
# Long when weekly trend is up and price breaks above daily Donchian(20) high with volume confirmation.
# Short when weekly trend is down and price breaks below daily Donchian(20) low with volume confirmation.
# Designed for low trade frequency (<50 total daily trades) to minimize fee drag.
# Works in bull/bear markets by following weekly trend while using daily breakouts for entry timing.

name = "1D_1W_Donchian_Breakout_Trend_Filter"
timeframe = "1d"
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
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian(10) for trend direction
    donchian_high_1w = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2
    
    # Weekly trend: price above mid = up, below mid = down
    trend_up = close_1w > donchian_mid_1w
    trend_down = close_1w < donchian_mid_1w
    
    # Align weekly trend to daily timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down)
    
    # Daily Donchian(20) for entry/exit
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(trend_up_aligned[i]) or
            np.isnan(trend_down_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly trend up + price breaks above daily Donchian high + volume spike
            if (trend_up_aligned[i] and 
                close[i] > donchian_high[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly trend down + price breaks below daily Donchian low + volume spike
            elif (trend_down_aligned[i] and 
                  close[i] < donchian_low[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below daily Donchian low OR weekly trend turns down
            if (close[i] < donchian_low[i]) or \
               not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above daily Donchian high OR weekly trend turns up
            if (close[i] > donchian_high[i]) or \
               not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals