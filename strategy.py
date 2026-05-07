#!/usr/bin/env python3
name = "12h_PriceAction_1dTrend_VolumeFilter"
timeframe = "12h"
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
    
    # 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume spike detection: current volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # 12-period high/low for breakout detection
    high_12 = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_12 = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 12)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(high_12[i]) or np.isnan(low_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12-period high + volume spike + above 1d EMA200
            if close[i] > high_12[i] and volume_spike[i] and close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12-period low + volume spike + below 1d EMA200
            elif close[i] < low_12[i] and volume_spike[i] and close[i] < ema_200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price reverses back through 12-period level or trend changes
            if position == 1:
                if close[i] < low_12[i] or close[i] < ema_200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_12[i] or close[i] > ema_200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals