#!/usr/bin/env python3
name = "6h_Donchian_WeeklyPivot_Bias_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend: price above/below 20-period EMA (responsive trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    trend_up = close > ema_1w_aligned
    
    # Daily pivot: standard pivot points (PP, R1, S1, R2, S2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Donchian channel (20 periods) on 6h
    donchian_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_dn = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(donchian_up[i]) or np.isnan(donchian_dn[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > Donchian upper + weekly uptrend + price > R1 (bullish bias)
            if close[i] > donchian_up[i] and trend_up[i] and close[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower + weekly downtrend + price < S1 (bearish bias)
            elif close[i] < donchian_dn[i] and not trend_up[i] and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < Donchian lower or weekly trend down
            if close[i] < donchian_dn[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > Donchian upper or weekly trend up
            if close[i] > donchian_up[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals