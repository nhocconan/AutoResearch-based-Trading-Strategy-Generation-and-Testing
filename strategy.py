# [Experiment #159035] 6H_DONCHIAN_BREAKOUT_WEEKLY_PIVOT_VOLUME
# Hypothesis: 6-hour Donchian breakout with weekly pivot bias and daily volume spike.
# Long when price breaks above 6h Donchian high + price above weekly pivot + volume spike.
# Short when price breaks below 6h Donchian low + price below weekly pivot + volume spike.
# Weekly pivot acts as regime filter: only trade in direction of weekly bias.
# Volume spike filters false breakouts. Targets institutional participation.
# Designed for ~15-30 trades/year on 6h to avoid fee drag.
# Works in bull/bear by requiring volume confirmation and weekly bias.

name = "6H_DONCHIAN_BREAKOUT_WEEKLY_PIVOT_VOLUME"
timeframe = "6h"
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
    
    # 6h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Weekly pivot point from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: P = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Daily volume spike confirmation (volume > 1.5x 20-day average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / vol_ma_20d
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above Donchian high + above weekly pivot + volume spike
            if (high[i] > donchian_high[i] and
                close[i] > weekly_pivot_aligned[i] and
                vol_spike_aligned[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + below weekly pivot + volume spike
            elif (low[i] < donchian_low[i] and
                  close[i] < weekly_pivot_aligned[i] and
                  vol_spike_aligned[i] > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (trend reversal)
            if low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (trend reversal)
            if high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals