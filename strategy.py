#!/usr/bin/env python3
name = "4h_ExponentialTrend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # DAILY TREND: 34-period EMA on daily closes (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4H MOMENTUM: 13-period EMA on 4h closes (entry signal)
    ema13_4h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # VOLUME CONFIRMATION: 20-period volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_4h[i]) or np.isnan(ema13_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 4h EMA13 above daily EMA34 + volume spike
            if (ema13_4h[i] > ema34_1d_4h[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 4h EMA13 below daily EMA34 + volume spike
            elif (ema13_4h[i] < ema34_1d_4h[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: 4h EMA13 crosses below daily EMA34
            if ema13_4h[i] < ema34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 4h EMA13 crosses above daily EMA34
            if ema13_4h[i] > ema34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals