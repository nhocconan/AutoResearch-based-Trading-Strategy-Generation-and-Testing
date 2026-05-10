#!/usr/bin/env python3
"""
1h_4hTrend_1dVolumeBreakout
Hypothesis: Use 4h EMA trend as directional filter, enter on 1h Donchian breakout with volume confirmation.
1d volume spike filter ensures participation from larger timeframe participants.
Designed for 15-30 trades/year on 1h timeframe to avoid fee drag while capturing trending moves.
Works in both bull and bear markets via trend filter - only trades in direction of 4h trend.
"""

name = "1h_4hTrend_1dVolumeBreakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend direction
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume EMA20 for volume spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d, additional_delay_bars=0)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current 1h volume > 1.5x 1d average volume (aligned)
    volume_spike = volume > (vol_ema_20_1d_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA (20), 1d volume EMA (20), Donchian (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_ema_20_1d_aligned[i]) or
            np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend AND price breaks above 20-period high with volume spike
            if close[i] > ema_20_4h_aligned[i] and high[i] > high_max_20[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend AND price breaks below 20-period low with volume spike
            elif close[i] < ema_20_4h_aligned[i] and low[i] < low_min_20[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 20-period low OR 4h trend turns down
            if low[i] < low_min_20[i] or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 20-period high OR 4h trend turns up
            if high[i] > high_max_20[i] or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals