#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wTrend_Filter
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun cross acts as momentum signal, 
filtered by weekly trend (price above/below weekly Kumo cloud) to avoid counter-trend 
whipsaws. Works in bull markets (TK cross up in uptrend) and bear markets (TK cross down in downtrend).
Uses discrete position sizing (0.25) to limit fee drag and drawdown. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    period9_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    period26_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    period52_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    return tenkan_sen, kijun_sen, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for Ichimoku (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1w data for weekly trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1d Ichimoku
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align 1d Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Weekly trend: price above/below weekly cloud
    # For simplicity, use weekly close vs weekly EMA20 as trend proxy
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 6h volume filter: current volume > 1.5 * 20-period volume MA
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Ichimoku warmup (52) + volume MA (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Ichimoku TK cross signals
        tk_cross_up = (tenkan_1d_aligned[i] > kijun_1d_aligned[i]) and \
                      (tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1])
        tk_cross_down = (tenkan_1d_aligned[i] < kijun_1d_aligned[i]) and \
                        (tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1])
        
        # Weekly trend filter: price above/below weekly EMA20
        weekly_uptrend = curr_close > ema_20_1w_aligned[i]
        weekly_downtrend = curr_close < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: TK cross up + weekly uptrend + volume filter
            if tk_cross_up and weekly_uptrend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + weekly downtrend + volume filter
            elif tk_cross_down and weekly_downtrend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: TK cross down or weekly trend turns down
            if tk_cross_down or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up or weekly trend turns up
            if tk_cross_up or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0