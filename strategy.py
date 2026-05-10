#!/usr/bin/env python3
# 4h_Donchian_Breakout_1dTrend_Volume_Confirmation
# Hypothesis: 4-hour Donchian channel (20-period) breakouts filtered by 1-day EMA trend (50-period) and volume spikes.
# The strategy captures strong trending moves while avoiding whipsaws in choppy markets by requiring alignment with
# higher-timeframe trend and elevated volume. Designed for 4h timeframe to balance trade frequency and signal quality.
# Works in both bull and bear markets by following the 1-day trend direction. Target: 20-50 trades/year.

name = "4h_Donchian_Breakout_1dTrend_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period high/low) - calculated on 4h data then aligned
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d EMA trend filter (50-period) - calculated on 1d data then aligned
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high, 1d EMA uptrend, volume confirmation, session active
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low, 1d EMA downtrend, volume confirmation, session active
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 4h Donchian low OR 1d EMA turns down
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above 4h Donchian high OR 1d EMA turns up
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals