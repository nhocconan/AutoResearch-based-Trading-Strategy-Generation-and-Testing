#!/usr/bin/env python3
"""
12h_Camarilla_R1_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On 12h timeframe, breakouts at weekly Camarilla R1 (long) and S3 (short) levels with volume confirmation and 1w trend alignment capture momentum in both bull and bear markets. Weekly trend filter ensures directional bias, while volume spike confirms institutional interest. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "12h_Camarilla_R1_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Camarilla pivot levels from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    range_1w = high_1w - low_1w
    # S3 = close - (range * 1.25)
    # R1 = close + (range * 1.0833)
    s3 = close_1w - (range_1w * 1.25000)
    r1 = close_1w + (range_1w * 1.08333)
    
    # Align to 12h timeframe (wait for weekly bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Weekly trend filter: EMA 34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 2.0x 40-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s3_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_34_1w_aligned[i]
        is_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume spike + weekly uptrend
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 + volume spike + weekly downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S3 (opposite side)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals