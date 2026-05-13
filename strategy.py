#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from 1-day combined with 1-day trend filter and volume confirmation on 12h timeframe.
Breakout above R1 with 1-day uptrend and volume spike = long.
Breakdown below S1 with 1-day downtrend and volume spike = short.
Exit at opposite Camarilla level (S1 for long, R1 for short) or when 1-day trend reverses.
Uses weekly trend filter for higher timeframe bias in bear markets.
Target: 12-37 trades/year per symbol.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Camarilla levels (calculate from prior 1-day bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1-day OHLC
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla calculations
    R1 = pc + 1.1 * (ph - pl) / 12
    S1 = pc - 1.1 * (ph - pl) / 12
    
    # Align Camarilla levels to 12h timeframe (wait for 1-day bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1-day trend filter: EMA50
    ema_50_1d = pd.Series(pc).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = pc > ema_50_1d
    downtrend_1d = pc < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Weekly trend filter (for bear market bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
        uptrend_1w = df_1w['close'].values > ema_50_1w
        downtrend_1w = df_1w['close'].values < ema_50_1w
        uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
        downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    else:
        uptrend_1w_aligned = np.ones(n, dtype=bool)
        downtrend_1w_aligned = np.ones(n, dtype=bool)
    
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
        uptrend_1d_val = uptrend_1d_aligned[i]
        downtrend_1d_val = downtrend_1d_aligned[i]
        uptrend_1w_val = uptrend_1w_aligned[i]
        downtrend_1w_val = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 1-day uptrend, weekly uptrend filter, volume confirmation
            if close[i] > r1 and uptrend_1d_val and uptrend_1w_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 1-day downtrend, weekly downtrend filter, volume confirmation
            elif close[i] < s1 and downtrend_1d_val and downtrend_1w_val and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 1-day trend turns down
            if close[i] < s1 or not uptrend_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or 1-day trend turns up
            if close[i] > r1 or not downtrend_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals