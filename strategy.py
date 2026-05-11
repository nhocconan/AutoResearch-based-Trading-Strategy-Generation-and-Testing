#!/usr/bin/env python3
name = "12h_Donchian_20_20_Trend_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # 12h Donchian(20) channels
    period = 20
    high_max = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_min = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1w trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    # Session filter: 08-20 UTC (for 12h bars, use start time)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian, above both EMAs, volume, session
            if close[i] > high_max[i] and close[i] > ema_34_1d_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian, below both EMAs, volume, session
            elif close[i] < low_min[i] and close[i] < ema_34_1d_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below lower Donchian or below either EMA
            if close[i] < low_min[i] or close[i] < ema_34_1d_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above upper Donchian or above either EMA
            if close[i] > high_max[i] or close[i] > ema_34_1d_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals