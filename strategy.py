#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter.
- Entry: Donchian(20) breakout in direction of 1d ADX>25 trend (long if +DI>-DI, short if -DI>+DI)
- Confirm: 1d volume > 1.5x 20-period average
- Exit: Opposite Donchian(10) breakout or ADX<20 (range regime)
- Position size: 0.25 discrete level
- Works in bull/bear via ADX trend filter and volume confirmation
- Target: 20-50 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20 for entry, 10 for exit)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # 1d ADX for trend regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align to 1d index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM
    tr_period = 14
    tr_sum = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_dm_sum = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm_sum = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # 1d volume spike
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # ADX, Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or
            np.isnan(minus_di_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(high_10[i]) or
            np.isnan(low_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = df_1d['volume'].values[np.searchsorted(df_1d.index.values, prices['open_time'].values[i]) - 1] > 1.5 * vol_ma_aligned[i] if i > 0 and np.searchsorted(df_1d.index.values, prices['open_time'].values[i]) > 0 else False
        
        # Trend direction from ADX
        is_uptrend = adx_aligned[i] > 25 and plus_di_aligned[i] > minus_di_aligned[i]
        is_downtrend = adx_aligned[i] > 25 and minus_di_aligned[i] > plus_di_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > high_20[i-1]  # Close above prior 20-period high
        breakout_down = close[i] < low_20[i-1]  # Close below prior 20-period low
        
        if position == 0:
            # Long: Donchian breakout up AND uptrend AND volume confirmation
            if breakout_up and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND downtrend AND volume confirmation
            elif breakout_down and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Opposite Donchian breakout OR ADX<20 (range)
            if close[i] < low_10[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Opposite Donchian breakout OR ADX<20 (range)
            if close[i] > high_10[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dADX_VolumeSpike"
timeframe = "4h"
leverage = 1.0