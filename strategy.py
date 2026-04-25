#!/usr/bin/env python3
"""
1d_Weekly_HTF_Trend_Long_Only_v1
Hypothesis: On daily timeframe, go long when price breaks above weekly Donchian(20) high during a confirmed weekly uptrend (price > weekly EMA34), with volume spike confirmation (>1.5x 20-day avg volume). Exit when price breaks below weekly Donchian(20) low or weekly trend turns bearish. This captures medium-term trend continuation with minimal trades (~15-25/year) to avoid fee drag. Long-only design works in bull markets; flat in bear/range reduces drawdown. Tested on BTC/ETH/SOL with proper HTF alignment via mtf_data.
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
    
    # Get weekly data for HTF trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Weekly Donchian(20) channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: max(high_1w, window=20)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    
    # Donchian low: min(low_1w, window=20)
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: 1.5x 20-day average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start index: need warmup for weekly EMA34 and Donchian
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0 if position == 0 else 0.25
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high with volume spike and weekly uptrend
            long_signal = (close[i] > donchian_high_aligned[i] and 
                          volume_spike[i] and 
                          close[i] > ema_34_aligned[i])
            if long_signal:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Exit: price breaks below weekly Donchian low OR weekly trend turns bearish
            exit_signal = (close[i] < donchian_low_aligned[i] or 
                          close[i] < ema_34_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_HTF_Trend_Long_Only_v1"
timeframe = "1d"
leverage = 1.0