#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts with 1d trend filter and volume spikes capture breakouts in both bull and bear markets.
Long when price breaks above R3 with 1d uptrend and volume spike; short when breaks below S3 with 1d downtrend and volume spike.
Exit on opposite Camarilla level touch or trend reversal. Target: 25-40 trades/year per symbol.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # 4-day high/low for Camarilla calculation (use 4 prior days to avoid lookahead)
    high_4d = np.zeros(n)
    low_4d = np.zeros(n)
    for i in range(n):
        if i < 96:  # Need 4 days of 4h data (4*24=96)
            high_4d[i] = np.nan
            low_4d[i] = np.nan
        else:
            high_4d[i] = np.max(high[i-96:i])
            low_4d[i] = np.min(low[i-96:i])
    
    # Camarilla levels: R3/S3 based on prior 4-day range
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    for i in range(n):
        if np.isnan(high_4d[i]) or np.isnan(low_4d[i]):
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
        else:
            range_4d = high_4d[i] - low_4d[i]
            camarilla_r3[i] = close[i-1] + range_4d * 1.1 / 4  # R3 = C + 1.1*(H-L)/4
            camarilla_s3[i] = close[i-1] - range_4d * 1.1 / 4  # S3 = C - 1.1*(H-L)/4
    
    # Volume confirmation: volume > 2.0 * 24-period average (4 days)
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 2.0 * vol_ma
    
    # 1d trend: EMA50 on daily
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(96, n):  # Start after 4 days of data
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R3, 1d uptrend, volume confirmation
            if not np.isnan(r3) and close[i] > r3 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3, 1d downtrend, volume confirmation
            elif not np.isnan(s3) and close[i] < s3 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S3 or 1d trend turns down
            if not np.isnan(s3) and close[i] < s3 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R3 or 1d trend turns up
            if not np.isnan(r3) and close[i] > r3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals