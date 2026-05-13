#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_TrendVolume
Hypothesis: Camarilla pivot levels (R1/S1) from 1d combined with 12h trend (EMA50) and volume confirmation provide robust entries in both bull and bear markets. 
Breakout above R1 with uptrend and volume spike = long. Breakdown below S1 with downtrend and volume spike = short. 
Exit on opposite touch (S1 for long, R1 for short) or trend reversal. Uses 1w trend filter for higher timeframe bias.
Target: 12-37 trades/year per symbol.
"""

name = "12h_Camarilla_Pivot_R1S1_Breakout_TrendVolume"
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
    
    # 12h EMA50 for trend
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = close > ema_50
    downtrend_12h = close < ema_50
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Daily OHLC for Camarilla pivots (use prior day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1: (H+L+C)/3 ± 1.1*(H-L)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = pivot + 1.1 * (high_1d - low_1d) / 12
    s1 = pivot - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 12h timeframe (these levels are valid for the entire day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        uptrend = uptrend_12h[i]
        downtrend = downtrend_12h[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 12h uptrend, 1w uptrend filter, volume confirmation
            if close[i] > r1_val and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 12h downtrend, 1w downtrend filter, volume confirmation
            elif close[i] < s1_val and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 12h trend turns down
            if close[i] < s1_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or 12h trend turns up
            if close[i] > r1_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals