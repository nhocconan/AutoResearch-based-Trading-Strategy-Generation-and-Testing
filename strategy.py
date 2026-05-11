#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
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
    
    # 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 0.1042)  # R1 = close + (range * 1.1/10.5)
    s1_1d = close_1d - (range_1d * 0.1042)  # S1 = close - (range * 1.1/10.5)
    
    # Align levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (4h volume > 2.0x 24-period average)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 2.0 * volume_ma24
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1, above EMA34, volume confirmation, session
            if close[i] > r1_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S1, below EMA34, volume confirmation, session
            elif close[i] < s1_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below EMA34
            if close[i] < s1_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Close above R1 or above EMA34
            if close[i] > r1_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals