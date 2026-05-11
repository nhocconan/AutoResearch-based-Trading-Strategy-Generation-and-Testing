#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # 12h trend: close above/below 12h EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    trend_up = close > ema_12h_aligned
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma20_12h)
    volume_filter = volume > 1.5 * vol_ma20_12h_aligned
    
    # 1d Camarilla levels (R1, S1) from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla R1 and S1 calculation
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma20_12h_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R1 + 12h uptrend + volume filter
            if close[i] > camarilla_r1_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S1 + 12h downtrend + volume filter
            elif close[i] < camarilla_s1_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S1 or 12h trend down
            if close[i] < camarilla_s1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R1 or 12h trend up
            if close[i] > camarilla_r1_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals