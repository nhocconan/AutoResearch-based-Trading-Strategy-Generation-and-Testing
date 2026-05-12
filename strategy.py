#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Camarilla levels (pivot-based)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r3 = close_1d + (range_hl * 1.1 / 4)
    r4 = close_1d + (range_hl * 1.1 / 2)
    s3 = close_1d - (range_hl * 1.1 / 4)
    s4 = close_1d - (range_hl * 1.1 / 2)
    
    # Align to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # ensure EMA34 data ready
    
    for i in range(start_idx, n):
        # Skip if Camarilla data not ready
        if np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_4h[i]) and (close[i] > ema34_4h[i]) and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_4h[i]) and (close[i] < ema34_4h[i]) and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below S3 or below 1d EMA34
            if (close[i] < s3_4h[i]) or (close[i] < ema34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above R3 or above 1d EMA34
            if (close[i] > r3_4h[i]) or (close[i] > ema34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals