#!/usr/bin/env python3
"""
1h_4h_Donchian_1d_Trend_Breakout_v1
Hypothesis: Combines 4h Donchian breakout with 1-day trend filter for direction,
using 1h only for precise entry timing and volume confirmation. Targets 15-37 trades/year
by requiring multiple confluence factors to minimize overtrading. Works in bull/bear via
trend filter and volatility-based breakout levels.
"""

name = "1h_4h_Donchian_1d_Trend_Breakout_v1"
timeframe = "1h"
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
    
    # 4h Donchian channels (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1-day trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high AND above 1-day EMA with volume confirmation
            if close[i] > donchian_high_aligned[i] and close[i] > ema_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low AND below 1-day EMA with volume confirmation
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price breaks below 4h Donchian low OR below 1-day EMA
            if close[i] < donchian_low_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above 4h Donchian high OR above 1-day EMA
            if close[i] > donchian_high_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals