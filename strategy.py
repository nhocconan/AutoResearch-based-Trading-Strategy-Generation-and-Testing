#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Daily trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    daily_uptrend = close > ema_34_1d_aligned
    
    # Daily Camarilla levels: use previous day's OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    range_ = daily_high - daily_low
    R3 = daily_close + 1.1 * range_ / 12
    S3 = daily_close - 1.1 * range_ / 12
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: 2x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        if np.isnan(daily_uptrend[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3, daily uptrend, volume spike
            if close[i] > R3_aligned[i] and daily_uptrend[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3, daily downtrend, volume spike
            elif close[i] < S3_aligned[i] and not daily_uptrend[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price re-enters below R3 or daily trend flips
            if close[i] < R3_aligned[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price re-enters above S3 or daily trend flips
            if close[i] > S3_aligned[i] or daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals