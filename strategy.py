#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_Trend_Volume
Hypothesis: 12h breakouts of Camarilla R1/S1 levels with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakout above R1 with 1d uptrend and volume spike = long.
Breakdown below S1 with 1d downtrend and volume spike = short.
Exit on opposite level touch or trend reversal. Uses 1w trend filter for higher timeframe bias.
Target: 12-37 trades/year per symbol.
"""

name = "12h_1d_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "12h"
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
    
    # Calculate 1d Camarilla levels (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar high, low, close
    ph = df_1d['high'].shift(1).values  # previous day high
    pl = df_1d['low'].shift(1).values   # previous day low
    pc = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla R1 and S1
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # 1w trend filter (additional bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 2.0 * 24-period average (24*12h = 12 days)
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # warmup for 24-period vol MA and EMA50
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        uptrend_1d_val = uptrend_1d_aligned[i]
        downtrend_1d_val = downtrend_1d_aligned[i]
        uptrend_1w_val = uptrend_1w_aligned[i]
        downtrend_1w_val = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 1d uptrend, 1w uptrend filter, volume confirmation
            if close[i] > r1_val and uptrend_1d_val and uptrend_1w_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 1d downtrend, 1w downtrend filter, volume confirmation
            elif close[i] < s1_val and downtrend_1d_val and downtrend_1w_val and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 1d trend turns down
            if close[i] < s1_val or not uptrend_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or 1d trend turns up
            if close[i] > r1_val or not downtrend_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals