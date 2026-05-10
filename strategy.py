#!/usr/bin/env python3
# 1h_4h_1d_PriceAction_Volume_Breakout
# Hypothesis: Capture breakouts from 4-hour price consolidation (high-low range) confirmed by
# 1-day trend (EMA50) and volume spikes, executed on 1h for timing. Works in bull/bear by
# following higher-timeframe trend. Target: 15-30 trades/year (~60-120 total) to avoid fee drag.

name = "1h_4h_1d_PriceAction_Volume_Breakout"
timeframe = "1h"
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
    
    # 4h consolidation range (20-period high-low range)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    range_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    range_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    range_high_aligned = align_htf_to_ltf(prices, df_4h, range_high)
    range_low_aligned = align_htf_to_ltf(prices, df_4h, range_low)
    
    # 1d EMA trend filter (50-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(range_high_aligned[i]) or np.isnan(range_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h range high, 1d EMA uptrend, volume confirmation, session active
            if (close[i] > range_high_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h range low, 1d EMA downtrend, volume confirmation, session active
            elif (close[i] < range_low_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 4h range low OR 1d EMA turns down
            if (close[i] < range_low_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price breaks above 4h range high OR 1d EMA turns up
            if (close[i] > range_high_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals