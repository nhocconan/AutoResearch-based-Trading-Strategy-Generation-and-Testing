#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
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
    
    # Daily data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    # Weekly pivot from previous week (7 days)
    week_high = df_1d['high'].rolling(window=7, min_periods=7).max().shift(1).values
    week_low = df_1d['low'].rolling(window=7, min_periods=7).min().shift(1).values
    week_close = df_1d['close'].shift(1).values
    
    weekly_pivot = (week_high + week_low + week_close) / 3
    r1 = 2 * weekly_pivot - week_low
    s1 = 2 * weekly_pivot - week_high
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend filter: price above/below 20-day EMA
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 6h Donchian channel (20 periods = 5 days)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and EMA need 20 periods
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout + above weekly R1 + above daily EMA20 + volume
            if (close[i] > donch_high[i] and 
                close[i] > r1_aligned[i] and 
                close[i] > ema20_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + below weekly S1 + below daily EMA20 + volume
            elif (close[i] < donch_low[i] and 
                  close[i] < s1_aligned[i] and 
                  close[i] < ema20_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Donchian break in opposite direction or return to weekly pivot
            if position == 1:
                if close[i] < donch_low[i] or close[i] < weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_high[i] or close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals