#!/usr/bin/env python3
# 1h_Turtle_Trader_4hTrend_VolumeFilter
# Hypothesis: Turtle-style breakouts on 1h with 4h trend filter and volume confirmation.
# Uses 4h Donchian channels for trend direction and 1h Donchian breakouts for entry.
# Volume filter (>1.5x average) reduces false breakouts. Works in bull via long breakouts
# and bear via short breakdowns. Target: 15-37 trades/year (60-150 over 4 years).
# Position size 0.20 keeps drawdown manageable during 2022 crash.

name = "1h_Turtle_Trader_4hTrend_VolumeFilter"
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
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian(20) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # 1h Donchian(20) for entry signals
    donchian_high_20_1h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume / 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 > 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian calculations
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 4h Donchian
        # Uptrend: price above 4h 20-period high
        # Downtrend: price below 4h 20-period low
        uptrend = close[i] > donchian_high_20_aligned[i]
        downtrend = close[i] < donchian_low_20_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above 1h 20-period high + uptrend + volume
            if close[i] > donchian_high_20_1h[i-1] and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 1h 20-period low + downtrend + volume
            elif close[i] < donchian_low_20_1h[i-1] and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price breaks below 1h 20-period low or trend ends
            if close[i] < donchian_low_20_1h[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above 1h 20-period high or trend ends
            if close[i] > donchian_high_20_1h[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals