#!/usr/bin/env python3
name = "1h_4h1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1h"
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
    
    # Session filter: 8-20 UTC (pre-compute hours)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Previous day's Camarilla levels (R3, S3)
    df_1d_prev = df_1d.copy()
    df_1d_prev['high_prev'] = df_1d_prev['high'].shift(1)
    df_1d_prev['low_prev'] = df_1d_prev['low'].shift(1)
    df_1d_prev['close_prev'] = df_1d_prev['close'].shift(1)
    
    # Calculate Camarilla levels for current day based on previous day
    high_prev = df_1d_prev['high_prev'].values
    low_prev = df_1d_prev['low_prev'].values
    close_prev = df_1d_prev['close_prev'].values
    
    # Camarilla R3 and S3
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    R3_aligned = align_htf_to_ltf(prices, df_1d_prev, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d_prev, S3)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter: only trade during 8-20 UTC
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R3 + 1d uptrend + volume spike
            if close[i] > R3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close < S3 + 1d downtrend + volume spike
            elif close[i] < S3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close < S3 or 1d trend down
            if close[i] < S3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close > R3 or 1d trend up
            if close[i] > R3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals