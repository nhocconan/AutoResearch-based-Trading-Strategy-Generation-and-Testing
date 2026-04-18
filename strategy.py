#!/usr/bin/env python3
"""
4h_Donchian20_24hVolumeSurge_TrendFilter
Hypothesis: Breakouts of 4h Donchian channel (20-period) combined with 24h volume surge
and 1d EMA trend filter capture strong momentum moves in both bull and bear markets.
The 24h volume surge ensures institutional participation, while EMA filter avoids
counter-trend trades. Targets 20-40 trades/year with position size 0.25.
"""

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
    
    # 4h Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False).values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 24h volume MA (96 periods of 15m = 24h)
    vol_ma_24h = np.full(n, np.nan)
    for i in range(96, n):
        vol_ma_24h[i] = np.mean(volume[i-96:i])
    
    # Volume surge: current volume > 2.0 * 24h average
    volume_surge = np.zeros(n, dtype=bool)
    for i in range(96, n):
        if vol_ma_24h[i] > 0:
            volume_surge[i] = volume[i] > 2.0 * vol_ma_24h[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 96)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24h[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high + volume surge + price above 1d EMA34
            if close[i] > donchian_high[i] and volume_surge[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume surge + price below 1d EMA34
            elif close[i] < donchian_low[i] and volume_surge[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_24hVolumeSurge_TrendFilter"
timeframe = "4h"
leverage = 1.0