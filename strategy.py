#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
Hypothesis: Combines Camarilla pivot levels (R1/S1) with 12h EMA trend and volume confirmation for high-probability breakout trades.
Designed for low trade frequency (20-40/year) with clear entry/exit rules. Works in both bull and bear markets by trading breakouts in the direction of higher timeframe trend.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
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
    
    # Calculate 12h EMA trend (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    # Typical price for pivot calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range
    range_hl = df_1d['high'] - df_1d['low']
    # Camarilla levels
    r1 = pivot + (range_hl * 1.0 / 12)
    s1 = pivot - (range_hl * 1.0 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above R1, 12h uptrend, volume confirmation
            if close[i] > r1_aligned[i] and ema_50_12h_aligned[i] > 0 and close[i] > ema_50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1, 12h downtrend, volume confirmation
            elif close[i] < s1_aligned[i] and ema_50_12h_aligned[i] > 0 and close[i] < ema_50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or 12h trend turns down
            if close[i] < s1_aligned[i] or (ema_50_12h_aligned[i] > 0 and close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or 12h trend turns up
            if close[i] > r1_aligned[i] or (ema_50_12h_aligned[i] > 0 and close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals