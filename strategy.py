#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Trend_Volume
Hypothesis: On 12h timeframe, price breaking above/below Camarilla R1/S1 levels from prior 1d,
confirmed by 1d trend and volume spike, provides high-probability trend-following entries.
Works in both bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).
Target: 15-25 trades/year per symbol.
"""

name = "12h_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(prices['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (vol_ma * 1.5)
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    rang = high_1d - low_1d
    camarilla_r1 = close_1d + (1.1 * rang / 12)
    camarilla_s1 = close_1d - (1.1 * rang / 12)
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d data to 12h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get aligned values
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: price breaks above R1 + 1d uptrend + volume spike
            if prices['close'].iloc[i] > r1 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + 1d downtrend + volume spike
            elif prices['close'].iloc[i] < s1 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below R1 or trend breaks
            if prices['close'].iloc[i] < r1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above S1 or trend breaks
            if prices['close'].iloc[i] > s1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals