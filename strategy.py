#!/usr/bin/env python3
# 1d_1w_Camarilla_R3_S3_Breakout_Volume
# Hypothesis: Uses weekly Camarilla pivot levels (R3, S3) from 1w timeframe for structure.
# Enters on 1d breakouts of these levels in the direction of the weekly trend (EMA34).
# Volume confirmation (>1.5x 20-period average) filters for institutional participation.
# Designed for low trade frequency (<150 total 1d trades) to minimize fee drag.
# Works in bull/bear markets by following the weekly trend while using 1d breaks for precise entries.

name = "1d_1w_Camarilla_R3_S3_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 1d timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for Camarilla pivot levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for weekly timeframe
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + (Range * 1.1)
    # S3 = Pivot - (Range * 1.1)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + (range_1w * 1.1)
    s3_1w = pivot_1w - (range_1w * 1.1)
    
    # Weekly trend: EMA34 of close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align weekly data to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if (np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or
            np.isnan(uptrend_1w_aligned[i]) or
            np.isnan(downtrend_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend on 1w + price breaks above R3 + volume spike
            if (uptrend_1w_aligned[i] and 
                close[i] > r3_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend on 1w + price breaks below S3 + volume spike
            elif (downtrend_1w_aligned[i] and 
                  close[i] < s3_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR 1w trend turns down
            if (close[i] < s3_1w_aligned[i]) or \
               downtrend_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR 1w trend turns up
            if (close[i] > r3_1w_aligned[i]) or \
               uptrend_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals