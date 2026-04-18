#!/usr/bin/env python3
"""
1h_4h_1d_Trend_Follow_with_Volume_and_Session_Filter
Hypothesis: Use 4h/1d trend direction (EMA cross) for signal direction, 1h for entry timing with volume confirmation, and session filter (08-20 UTC) to reduce noise. Designed to work in both bull and bear markets by following higher timeframe trends with confirmation.
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
    open_time = prices['open_time'].values
    
    # 4h EMA trend (fast/slow crossover)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_fast = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close_4h).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_slow)
    
    # 1d EMA for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: >1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(in_session[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_fast_val = ema_fast_aligned[i]
        ema_slow_val = ema_slow_aligned[i]
        ema_1d_val = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        session_ok = in_session[i]
        
        if position == 0:
            # Long: 4h EMA bullish + price above 1d EMA + volume + session
            if ema_fast_val > ema_slow_val and price > ema_1d_val and vol_spike and session_ok:
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA bearish + price below 1d EMA + volume + session
            elif ema_fast_val < ema_slow_val and price < ema_1d_val and vol_spike and session_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: 4h EMA bearish or price below 1d EMA
            if ema_fast_val < ema_slow_val or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: 4h EMA bullish or price above 1d EMA
            if ema_fast_val > ema_slow_val or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4h_1d_Trend_Follow_with_Volume_and_Session_Filter"
timeframe = "1h"
leverage = 1.0