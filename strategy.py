#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: Trade 4h Donchian(20) breakouts in direction of 1d ATR-based trend with volume confirmation.
ATR trend filter identifies strong trending markets (ADX-like) without lag. Donchian breakouts capture momentum.
Volume spike confirms institutional interest. Discrete sizing 0.25 to limit fee churn. Target 20-40 trades/year.
Works in bull (breakouts continue) and bear (strong trends persist) via ATR trend filter.
"""

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
    
    # Get daily data for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for trend strength
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily ATR ratio: current ATR / 50-period average ATR
    # High ratio = expanding volatility = trending market
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / (atr_ma_50_1d + 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR MA (50) and Donchian (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND ATR ratio > 1.2 (trending up) AND volume spike
            long_setup = (close[i] > donchian_high[i]) and \
                         (atr_ratio_1d_aligned[i] > 1.2) and \
                         volume_spike[i]
            # Short: price breaks below Donchian low AND ATR ratio > 1.2 (trending down) AND volume spike
            short_setup = (close[i] < donchian_low[i]) and \
                          (atr_ratio_1d_aligned[i] > 1.2) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR ATR ratio falls below 0.8 (trend weakening)
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (atr_ratio_1d_aligned[i] < 0.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR ATR ratio falls below 0.8 (trend weakening)
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (atr_ratio_1d_aligned[i] < 0.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0