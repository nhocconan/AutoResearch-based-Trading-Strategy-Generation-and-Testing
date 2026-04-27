#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend with volume confirmation.
Weekly pivot trend: price above/below weekly pivot point (PP) from prior week.
Long when price breaks above Donchian(20) high AND price > weekly PP AND volume spike.
Short when price breaks below Donchian(20) low AND price < weekly PP AND volume spike.
Exit on opposite Donchian breakout or loss of weekly pivot alignment.
Designed for 12-30 trades/year on 6h to minimize fee drag while capturing strong directional moves aligned with weekly structure.
Works in bull markets (breakouts with weekly uptrend) and bear markets (breakdowns with weekly downtrend).
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
    
    # Calculate 1d data ONCE before loop for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    # Weekly pivot from prior week: need 5 daily bars (1 week)
    # We'll use rolling window of 5 on 1d data to get weekly OHLC
    df_1d_close = pd.Series(df_1d['close'].values)
    df_1d_high = pd.Series(df_1d['high'].values)
    df_1d_low = pd.Series(df_1d['low'].values)
    
    # Weekly high, low, close (prior complete week)
    weekly_high = df_1d_high.rolling(window=5, min_periods=5).max().shift(1)  # shift to use prior week
    weekly_low = df_1d_low.rolling(window=5, min_periods=5).min().shift(1)
    weekly_close = df_1d_close.rolling(window=5, min_periods=5).mean().shift(1)
    
    # Weekly pivot point: (H+L+C)/3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp.values)
    
    # Donchian(20) on 6h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for weekly PP (5*1d = ~35 6h bars), Donchian(20), volume avg
    start_idx = max(35, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_pp_val = weekly_pp_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Donchian breakout with weekly pivot alignment and volume spike
            # Long: Close > Donchian high AND price > weekly PP AND volume spike
            # Short: Close < Donchian low AND price < weekly PP AND volume spike
            long_condition = (close_val > donch_high[i] and 
                            close_val > weekly_pp_val and 
                            vol_spike)
            short_condition = (close_val < donch_low[i] and 
                             close_val < weekly_pp_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Donchian low OR loses weekly PP alignment
            if close_val < donch_low[i] or close_val < weekly_pp_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian high OR loses weekly PP alignment
            if close_val > donch_high[i] or close_val > weekly_pp_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0