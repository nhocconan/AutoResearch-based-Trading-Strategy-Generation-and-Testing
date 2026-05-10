#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses Camarilla R3/S3 levels from 1d as breakout levels with 1d EMA34 trend filter and volume spike confirmation.
# Enters long when price breaks above R3 with volume spike and 1d EMA34 uptrend.
# Enters short when price breaks below S3 with volume spike and 1d EMA34 downtrend.
# Exits when price returns to the Camarilla center (P) level or trend reverses.
# Designed for 15-30 trades/year on 6h to avoid overtrading and work in both bull and bear markets.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Calculate volume spike: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    vol_ma[19] = np.mean(volume[0:20])
    for i in range(20, n):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_spike = volume > (vol_ma * 1.5)
    
    # 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla levels (R3, S3, P)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_p = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        range_1d = high_1d[i] - low_1d[i]
        camarilla_p[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        camarilla_r3[i] = camarilla_p[i] + (range_1d * 1.1 / 4)
        camarilla_s3[i] = camarilla_p[i] - (range_1d * 1.1 / 4)
    
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_p_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and 1d EMA34 uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and 1d EMA34 downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to P level or trend reversal (price below EMA)
            if close[i] < camarilla_p_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to P level or trend reversal (price above EMA)
            if close[i] > camarilla_p_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals