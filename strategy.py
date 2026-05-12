#!/usr/bin/env python3
# 4h_Donchian_20_Trend_Volume_Spike
# Hypothesis: Uses 4h Donchian channel breakouts aligned with 1d trend and volume confirmation.
# Enters long when price breaks above Donchian upper (20) with 1d uptrend and volume spike.
# Enters short when price breaks below Donchian lower (20) with 1d downtrend and volume spike.
# Uses 1d EMA50 as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee spike.
# Works in bull/bear markets by following 1d trend while using 4h Donchian breakouts for entries.

name = "4h_Donchian_20_Trend_Volume_Spike"
timeframe = "4h"
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
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h Donchian channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + 1d EMA50 uptrend + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + 1d EMA50 downtrend + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower OR closes below 1d EMA50
            if (close[i] < donchian_lower[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper OR closes above 1d EMA50
            if (close[i] > donchian_upper[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals