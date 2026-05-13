#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1_S1_Breakout_Trend_Volume
Hypothesis: On 1h timeframe, use 4h for trend direction and 1d for Camarilla pivot levels (R1, S1).
Breakout above 1d R1 with 4h uptrend and volume confirmation signals long.
Breakdown below 1d S1 with 4h downtrend and volume confirmation signals short.
This reduces trade frequency by requiring alignment across three timeframes and volume filter.
Target: 15-37 trades/year per symbol (60-150 total over 4 years).
Works in both bull and bear markets by following higher timeframe trend.
"""

name = "1h_4h_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1h"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    cam_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    cam_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h trend: 34 EMA
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_4h = close_4h > ema_34_4h
    downtrend_4h = close_4h < ema_34_4h
    
    # Align 4h trend to 1h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        # Get aligned values for current bar
        r1 = cam_r1_aligned[i]
        s1 = cam_s1_aligned[i]
        uptrend = uptrend_4h_aligned[i]
        downtrend = downtrend_4h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price breaks above 1d R1, 4h uptrend, volume confirmation
            if close[i] > r1 and uptrend and vol_conf:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below 1d S1, 4h downtrend, volume confirmation
            elif close[i] < s1 and downtrend and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below 1d S1 or 4h trend turns down
            if close[i] < s1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above 1d R1 or 4h trend turns up
            if close[i] > r1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals