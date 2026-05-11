#!/usr/bin/env python3
name = "1d_WilliamsAlligator_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA13
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    weekly_trend_up = close > ema13_1w_aligned
    
    # Williams Alligator (daily): SMA13(8), SMA8(5), SMA5(3)
    sma13 = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    sma8 = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    sma5 = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Williams Alligator signals: jaw (SMA13), teeth (SMA8), lips (SMA5)
    alligator_long = sma5 > sma8 and sma8 > sma13
    alligator_short = sma5 < sma8 and sma8 < sma13
    
    # Volume filter: daily volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(sma5[i]) or np.isnan(sma8[i]) or
            np.isnan(sma13[i]) or np.isnan(vol_ma20[i])):
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
            # Long: Alligator aligned up + weekly uptrend + volume filter
            if sma5[i] > sma8[i] and sma8[i] > sma13[i] and weekly_trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + weekly downtrend + volume filter
            elif sma5[i] < sma8[i] and sma8[i] < sma13[i] and not weekly_trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or weekly trend down
            if not (sma5[i] > sma8[i] and sma8[i] > sma13[i]) or not weekly_trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or weekly trend up
            if not (sma5[i] < sma8[i] and sma8[i] < sma13[i]) or weekly_trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals