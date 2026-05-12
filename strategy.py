#!/usr/bin/env python3
# 12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Filter
# Hypothesis: Uses daily Camarilla pivot levels (R3/S3) as structural support/resistance on 12h timeframe.
# Enters on breakouts of R3 (long) or S3 (short) only when aligned with weekly trend (EMA34).
# Weekly trend filter ensures we trade with the dominant higher timeframe momentum.
# Volume confirmation (>1.5x 20-period average) filters for institutional participation.
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
# Works in bull/bear markets by following weekly trend while using 12h breakouts for precise entries.

name = "12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Filter"
timeframe = "12h"
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
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng
    camarilla_s3 = close_1d - 1.1 * rng
    
    # Align daily Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_uptrend = close_1w > ema_34_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly uptrend + price breaks above R3 + volume spike
            if (weekly_uptrend_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price breaks below S3 + volume spike
            elif ((not weekly_uptrend_aligned[i]) and 
                  close[i] < s3_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR weekly trend turns down
            if (close[i] < s3_aligned[i]) or \
               (not weekly_uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR weekly trend turns up
            if (close[i] > r3_aligned[i]) or \
               weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals