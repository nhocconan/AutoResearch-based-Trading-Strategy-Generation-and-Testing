#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1d chop regime filter.
    # Donchian breakout captures volatility expansion after consolidation.
    # 1d volume spike (volume > 2.0 * 20-period MA) confirms institutional participation.
    # 1d chop regime (CHOP(14) between 38.2 and 61.8) ensures we avoid strong trends where breakouts fail.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 75-200 total trades over 4 years (19-50/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and chop regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA(20) for spike detection
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d True Range for chopiness index
    tr1 = pd.Series(high_1d).sub(pd.Series(low_1d)).abs()
    tr2 = pd.Series(high_1d).sub(pd.Series(close_1d).shift(1)).abs()
    tr3 = pd.Series(low_1d).sub(pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Chopiness Index (CHOP)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2.0 * 20-period MA
        volume_spike = volume_1d[i // 16] > 2.0 * volume_ma_1d_aligned[i] if i // 16 < len(volume_1d) else False
        
        # Chop regime filter: CHOP between 38.2 and 61.8 (ranging market)
        chop_filter = (chop_aligned[i] > 38.2) and (chop_aligned[i] < 61.8)
        
        # Donchian breakout conditions
        long_breakout = (close[i] > highest_20[i-1]) and volume_spike and chop_filter
        short_breakout = (close[i] < lowest_20[i-1]) and volume_spike and chop_filter
        
        # Exit conditions: price returns to midpoint of Donchian channel
        donchian_mid = (highest_20[i-1] + lowest_20[i-1]) / 2
        long_exit = close[i] < donchian_mid
        short_exit = close[i] > donchian_mid
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0