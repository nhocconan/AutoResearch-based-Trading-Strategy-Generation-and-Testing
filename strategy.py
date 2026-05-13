#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirmation
Hypothesis: Camarilla pivot levels (R1, S1) from the weekly chart act as strong support/resistance levels.
A breakout above R1 with 1w uptrend and volume confirmation signals a long entry.
A breakdown below S1 with 1w downtrend and volume confirmation signals a short entry.
This strategy uses higher timeframe (1w) trend to reduce whipsaw and improve trend following.
Target: 7-25 trades/year per symbol (30-100 total over 4 years).
"""

name = "1d_1w_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each weekly bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    cam_r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    cam_s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align Camarilla levels to daily timeframe (wait for weekly bar to close)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1w, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1w, cam_s1)
    
    # Weekly trend: 34 EMA
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align weekly trend to daily
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values for current bar
        r1 = cam_r1_aligned[i]
        s1 = cam_s1_aligned[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price breaks above R1, 1w uptrend, volume confirmation
            if close[i] > r1 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1, 1w downtrend, volume confirmation
            elif close[i] < s1 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or 1w trend turns down
            if close[i] < s1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or 1w trend turns up
            if close[i] > r1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals