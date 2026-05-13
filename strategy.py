#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1_S1_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels (R1/S1) on 4h/1d act as support/resistance with institutional relevance. 
Breakouts above R1 or below S1 with volume confirmation and 4h/1d trend alignment capture institutional flow.
Works in both bull/bear markets by following higher timeframe trend. Volume filter ensures breakout authenticity.
Target: 15-35 trades/year per symbol.
"""

name = "1h_4h1d_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "1h"
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
    
    # Calculate Camarilla levels for 4h and 1d
    def calculate_camarilla(h, l, c):
        """Returns (R1, S1) from Camarilla pivot"""
        pivot = (h + l + c) / 3.0
        range_ = h - l
        r1 = c + (range_ * 1.1 / 12)
        s1 = c - (range_ * 1.1 / 12)
        return r1, s1
    
    # 4h Camarilla
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    r1_4h, s1_4h = calculate_camarilla(h_4h, l_4h, c_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 1d Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    r1_1d, s1_1d = calculate_camarilla(h_1d, l_1d, c_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: 20-period SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h trend filter: EMA50
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = df_4h['close'].values > ema_50_4h
    downtrend_4h = df_4h['close'].values < ema_50_4h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        vol_ok = volume[i] > vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        if position == 0:
            # LONG: close > R1 (4h OR 1d) + volume + 4h uptrend
            if vol_ok and (
                (close[i] > r1_4h_aligned[i] and uptrend_4h_aligned[i]) or
                (close[i] > r1_1d_aligned[i] and uptrend_4h_aligned[i])
            ):
                signals[i] = 0.20
                position = 1
            # SHORT: close < S1 (4h OR 1d) + volume + 4h downtrend
            elif vol_ok and (
                (close[i] < s1_4h_aligned[i] and downtrend_4h_aligned[i]) or
                (close[i] < s1_1d_aligned[i] and downtrend_4h_aligned[i])
            ):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < S1 (4h OR 1d) or trend change
            if (close[i] < s1_4h_aligned[i] or close[i] < s1_1d_aligned[i] or 
                not uptrend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: close > R1 (4h OR 1d) or trend change
            if (close[i] > r1_4h_aligned[i] or close[i] > r1_1d_aligned[i] or 
                not downtrend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals