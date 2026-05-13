#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_1D_Trend_Volume
Hypothesis: Camarilla pivot R1/S1 breakouts with daily trend filter and volume confirmation.
Long on break above R1 with daily uptrend and volume spike. Short on break below S1 with daily downtrend and volume spike.
Exit on opposite pivot touch or trend reversal. Uses 1d pivot levels and EMA for trend filter.
Target: 20-50 trades/year per symbol.
"""

name = "4h_Camarilla_Pivot_R1_S1_Breakout_1D_Trend_Volume"
timeframe = "4h"
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
    
    # Camarilla pivot from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily trend: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, daily uptrend, volume confirmation
            if close[i] > r1 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, daily downtrend, volume confirmation
            elif close[i] < s1 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or daily trend turns down
            if close[i] < s1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or daily trend turns up
            if close[i] > r1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals