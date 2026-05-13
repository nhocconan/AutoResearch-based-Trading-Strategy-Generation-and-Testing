#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_R4_S4_Breakout_Trend_Volume
Hypothesis: Weekly and daily trends define market bias. Breakouts beyond Camarilla R4/S4
levels (extreme levels) with volume confirmation and aligned weekly/daily trend capture
strong momentum moves. Works in bull/bear by following higher timeframe trend.
Target: 12-37 trades/year per symbol.
"""

name = "6h_1w_1d_Camarilla_R4_S4_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for daily
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    cam_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    cam_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # Daily trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align daily trend to 6h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Weekly trend filter: 21 EMA on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    uptrend_1w = close_1w > ema_21_1w
    downtrend_1w = close_1w < ema_21_1w
    
    # Align weekly trend to 6s
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values
        r4 = cam_r4_aligned[i]
        s4 = cam_s4_aligned[i]
        uptrend_1d = uptrend_1d_aligned[i]
        downtrend_1d = downtrend_1d_aligned[i]
        uptrend_1w = uptrend_1w_aligned[i]
        downtrend_1w = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price breaks above R4, daily & weekly uptrend, volume confirmation
            if close[i] > r4 and uptrend_1d and uptrend_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S4, daily & weekly downtrend, volume confirmation
            elif close[i] < s4 and downtrend_1d and downtrend_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S4 or trend turns down
            if close[i] < s4 or not (uptrend_1d and uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R4 or trend turns up
            if close[i] > r4 or not (downtrend_1d and downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals