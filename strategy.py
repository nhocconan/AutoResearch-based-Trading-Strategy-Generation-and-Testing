#!/usr/bin/env python3
# 1h_Bollinger_Band_Bounce_Volume_Trend
# Hypothesis: In 1h timeframe, price tends to bounce off Bollinger Bands (20,2) with volume confirmation
# and 4h trend alignment. This mean-reversion strategy works in both bull and bear markets
# by trading reversals from extremes while respecting the higher timeframe trend.
# Uses Bollinger Bands for entry/exit, volume for confirmation, and 4h EMA for trend filter.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "1h_Bollinger_Band_Bounce_Volume_Trend"
timeframe = "1h"
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
    
    # Bollinger Bands (20, 2) on 1h
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + bb_std * bb_std_dev
    bb_lower = bb_middle - bb_std * bb_std_dev
    
    # 4h data for trend filter and session filtering
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation (24-period average = 1 day for 1h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 34)  # Need enough history for BB and EMA
    
    for i in range(start_idx, n):
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or \
           np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches/bounces off lower BB, above 4h EMA34, volume confirmation
            if close[i] <= bb_lower[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.20
                position = 1
            # Short: price touches/bounces off upper BB, below 4h EMA34, volume confirmation
            elif close[i] >= bb_upper[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reaches middle BB or crosses below 4h EMA34
            if close[i] >= bb_middle[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reaches middle BB or crosses above 4h EMA34
            if close[i] <= bb_middle[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals