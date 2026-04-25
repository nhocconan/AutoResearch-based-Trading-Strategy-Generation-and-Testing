#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Daily Donchian channel (20) breakouts with weekly EMA50 trend filter and volume spike capture strong momentum moves. Works in bull markets (breakouts continuation) and bear markets (breakdown continuation). Uses discrete position sizing (0.30) to limit fee drag and drawdown. Target: 20-50 trades/year on 1d timeframe.
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
    
    # 1d data for Donchian channels (20)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to LTF (1d)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Donchian (20) + volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)  # align_htf_to_ltf handles warmup internally
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and 1w uptrend
            long_breakout = (curr_close > high_20_aligned[i]) and vol_spike[i] and (curr_close > ema_50_1w_aligned[i])
            # Short: price breaks below lower Donchian with volume spike and 1w downtrend
            short_breakout = (curr_close < low_20_aligned[i]) and vol_spike[i] and (curr_close < ema_50_1w_aligned[i])
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
            elif short_breakout:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend turns down
            if (curr_close < low_20_aligned[i]) or (curr_close < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend turns up
            if (curr_close > high_20_aligned[i]) or (curr_close > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0