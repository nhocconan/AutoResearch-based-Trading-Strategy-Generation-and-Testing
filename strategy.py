#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_R4_S4_Breakout_Trend_Volume
Hypothesis: Weekly R4/S4 levels are strong breakout levels. A breakout above R4 with 1d uptrend and volume continuation signals a long entry. A breakdown below S4 with 1d downtrend and volume continuation signals a short entry. Uses weekly structure for trend direction and daily trend for entry filter. Designed to capture strong momentum moves in both bull and bear markets with tight entries to avoid overtrading.
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each weekly bar
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    cam_r4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    cam_s4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for weekly bar to close)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1w, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1w, cam_s4)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align daily trend to 6h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.3 * 20-period average (to allow continuation)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_cont = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values for current bar
        r4 = cam_r4_aligned[i]
        s4 = cam_s4_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_cont = volume_cont[i]
        
        if position == 0:
            # LONG: price breaks above R4, 1d uptrend, volume continuation
            if close[i] > r4 and uptrend and vol_cont:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S4, 1d downtrend, volume continuation
            elif close[i] < s4 and downtrend and vol_cont:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S4 or 1d trend turns down
            if close[i] < s4 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R4 or 1d trend turns up
            if close[i] > r4 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals