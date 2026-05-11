#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Dyn"
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
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 1d volume filter: volume > 1.8x 34-day average
    vol_1d = df_1d['volume'].values
    vol_ma34_1d = pd.Series(vol_1d).rolling(window=34, min_periods=34).mean().values
    vol_ma34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma34_1d)
    volume_filter = volume > 1.8 * vol_ma34_1d_aligned
    
    # Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    range_1d = high_1d - low_1d
    R3 = close_1d_prev + 1.1 * (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d_prev - 1.1 * (high_1d - low_1d) * 1.1 / 4
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma34_1d_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R3 + daily uptrend + volume filter
            if close[i] > R3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close below S3 + daily downtrend + volume filter
            elif close[i] < S3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Close below S3 or daily trend down
            if close[i] < S3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Close above R3 or daily trend up
            if close[i] > R3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals