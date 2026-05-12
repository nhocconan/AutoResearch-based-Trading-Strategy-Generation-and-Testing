#!/usr/bin/env python3
"""
6h_1W_Camarilla_R3S3_Breakout_TrendVol
Hypothesis: 6-hour breakouts from weekly Camarilla R3/S3 levels with weekly EMA20 trend filter and volume spike confirmation.
Only takes long when price breaks above R3 with volume spike and weekly uptrend, short when breaks below S3 with volume spike and weekly downtrend.
Uses weekly timeframe for structure and trend, reducing trade frequency while capturing major breaks. Weekly R3/S3 are strong support/resistance levels that often lead to continuation when broken with volume.
Designed to work in both bull and bear markets by following the weekly trend direction on breakouts.
"""

name = "6h_1W_Camarilla_R3S3_Breakout_TrendVol"
timeframe = "6h"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly Camarilla R3 and S3 from previous week
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    rang_1w = prev_high_1w - prev_low_1w
    R3_1w = prev_close_1w + 1.1 * rang_1w * 3.0 / 4
    S3_1w = prev_close_1w - 1.1 * rang_1w * 3.0 / 4
    
    # Align weekly levels to 6h timeframe
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(R3_1w_aligned[i]) or 
            np.isnan(S3_1w_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above weekly EMA20 (weekly uptrend)
            if (close[i] > R3_1w_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below weekly EMA20 (weekly downtrend)
            elif (close[i] < S3_1w_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous week's H-L range OR closes below weekly EMA20
            if (close[i] < R3_1w_aligned[i] and close[i] > S3_1w_aligned[i]) or \
               close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous week's H-L range OR closes above weekly EMA20
            if (close[i] < R3_1w_aligned[i] and close[i] > S3_1w_aligned[i]) or \
               close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals