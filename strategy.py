#!/usr/bin/env python3
"""
4h_Chaikin_Oscillator_Volume_Trend
Hypothesis: Chaikin Oscillator crossing zero with volume confirmation and 4h trend filter captures
institutional flow shifts in both bull and bear markets. Uses 1d trend as higher timeframe filter.
Target: 20-40 trades/year per signal to avoid fee drag.
"""

name = "4h_Chaikin_Oscillator_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    # Money Flow Volume
    mfv = mfm * volume
    
    # Chaikin Oscillator: (3-period EMA of MFV) - (10-period EMA of MFV)
    ema3 = pd.Series(mfv).ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = pd.Series(mfv).ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # 4h trend: EMA34
    ema34_4h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_4h = close > ema34_4h
    downtrend_4h = close < ema34_4h
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema34_1d
    downtrend_1d = df_1d['close'].values < ema34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average (avoid excessive signals)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Get values
        chaikin_now = chaikin[i]
        chaikin_prev = chaikin[i-1]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: Chaikin crosses above zero, 4h uptrend, 1d uptrend filter, volume confirmation
            if chaikin_now > 0 and chaikin_prev <= 0 and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin crosses below zero, 4h downtrend, 1d downtrend filter, volume confirmation
            elif chaikin_now < 0 and chaikin_prev >= 0 and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin crosses below zero or 4h trend turns down
            if chaikin_now < 0 and chaikin_prev >= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin crosses above zero or 4h trend turns up
            if chaikin_now > 0 and chaikin_prev <= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals