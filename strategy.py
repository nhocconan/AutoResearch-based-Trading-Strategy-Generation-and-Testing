#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) on 1d combined with 1w trend filter and volume confirmation
captures breakouts with low false signals. Designed for 12h timeframe to keep trades < 150 total over 4 years.
Uses 1w EMA20 trend filter and volume > 1.5x average to reduce false signals.
Target: 20-40 trades/year per symbol to avoid fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    # Typical price for Camarilla: (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    P = typical_price.values
    H = df_1d['high'].values
    L = df_1d['low'].values
    # Camarilla R1 and S1
    R1 = P + 1.1 * (H - L) / 12
    S1 = P - 1.1 * (H - L) / 12
    
    # Align to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Trend filter: 1w EMA20
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    uptrend = close > ema_20_1w_aligned
    downtrend = close < ema_20_1w_aligned
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if position == 0:
            # LONG: break above R1, uptrend, volume confirmation
            if close[i] > R1_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, downtrend, volume confirmation
            elif close[i] < S1_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below S1 or trend reverses
            if close[i] < S1_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above R1 or trend reverses
            if close[i] > R1_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals