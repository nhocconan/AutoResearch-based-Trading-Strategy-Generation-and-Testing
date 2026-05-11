#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Slim"
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
    
    # Daily trend: EMA34
    close_series = pd.Series(close)
    ema_34_1d_raw = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = align_htf_to_ltf(prices, df_1d, ema_34_1d_raw)
    daily_uptrend = close > ema_34_1d
    
    # Daily Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 4h volume > 1.5x 20-period MA
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # ensure EMA34 ready
    
    for i in range(start_idx, n):
        if np.isnan(daily_uptrend[i]) or np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R3, daily uptrend, volume
            if close[i] > r3_4h[i] and daily_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S3, daily downtrend, volume
            elif close[i] < s3_4h[i] and not daily_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close < S3 or daily downtrend
            if close[i] < s3_4h[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close > R3 or daily uptrend
            if close[i] > r3_4h[i] or daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals