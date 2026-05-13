#!/usr/bin/env python3
"""
4h_Trix_Trend_Volume_Signal
Hypothesis: TRIX (15) crossing zero with 12h EMA trend and volume confirmation captures medium-term momentum in both bull and bear markets.
Long when TRIX crosses above zero with 12h uptrend and volume spike.
Short when TRIX crosses below zero with 12h downtrend and volume spike.
Exit when TRIX crosses back through zero or trend reverses.
Target: 20-40 trades/year per symbol.
"""

name = "4h_Trix_Trend_Volume_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX: EMA(EMA(EMA(close,15),15),15) - 1 period percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change(1) * 100
    trix = trix_raw.fillna(0).values
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # TRIX zero cross
        trix_now = trix[i]
        trix_prev = trix[i-1]
        cross_up = trix_prev <= 0 and trix_now > 0
        cross_down = trix_prev >= 0 and trix_now < 0
        
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: TRIX crosses up through zero, 12h uptrend, volume confirmation
            if cross_up and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses down through zero, 12h downtrend, volume confirmation
            elif cross_down and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses down through zero or 12h trend turns down
            if cross_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses up through zero or 12h trend turns up
            if cross_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals