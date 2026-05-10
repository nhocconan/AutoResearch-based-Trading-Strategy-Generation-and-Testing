#!/usr/bin/env python3
"""
6h_ElderRay_Signal_1dTrend_Volume
Hypothesis: Use Elder Ray Index (bull/bear power) with 13-period EMA on 6h,
filtered by 1d EMA34 trend direction and volume spike. Works in bull/bear by
aligning with daily trend, avoiding counter-trend trades. Target: 15-35 trades/year.
"""

name = "6h_ElderRay_Signal_1dTrend_Volume"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, n):
            ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 1.8x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34, 20)  # EMA13 + EMA34 + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 0:
            # Long: Positive bull power AND above daily EMA34 AND volume spike
            if bull_power[i] > 0 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Negative bear power AND below daily EMA34 AND volume spike
            elif bear_power[i] < 0 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull power turns negative OR price below daily EMA34
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear power turns positive OR price above daily EMA34
            if bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals