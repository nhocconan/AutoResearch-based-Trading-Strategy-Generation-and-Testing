#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Trend
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 with 1d trend confirmation and volume spike provides high-probability trend-following entries. Camarilla levels act as support/resistance, and breakouts with volume indicate institutional interest. Works in both bull and bear markets by following the 1d trend. Target: 15-30 trades/year per symbol.
"""

name = "12h_1d_Camarilla_R1_S1_Breakout_Trend"
timeframe = "12h"
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
    
    # Calculate 12h Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We'll calculate daily levels and align to 12h
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for each day
    camarilla_r1 = np.zeros(len(df_1d))
    camarilla_s1 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        camarilla_r1[i] = c + (h - l) * 1.1 / 12
        camarilla_s1[i] = c - (h - l) * 1.1 / 12
    
    # Align Camarilla levels to 12h (1-day delay for previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    vol_series = pd.Series(volume)
    for i in range(20, n):
        vol_ma[i] = vol_series.iloc[i-20:i].mean()
    
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: price breaks above R1, 1d uptrend, volume spike
            if close[i] > r1 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1, 1d downtrend, volume spike
            elif close[i] < s1 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or 1d trend turns down
            if close[i] < s1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or 1d trend turns up
            if close[i] > r1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals