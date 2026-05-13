#!/usr/bin/env python3
"""
12h_Trix_Volume_Spike_Trend_Weekly
Hypothesis: TRIX momentum combined with volume spikes and weekly trend filters works in both bull and bear markets.
Long when TRIX crosses above zero with volume spike and weekly uptrend.
Short when TRIX crosses below zero with volume spike and weekly downtrend.
Exit when TRIX crosses back through zero or trend weakens.
Weekly trend filter reduces whipsaws in ranging markets.
Target: 15-35 trades/year per symbol.
"""

name = "12h_Trix_Volume_Spike_Trend_Weekly"
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
    volume = prices['volume'].values
    
    # TRIX: Triple EMA of log returns, period=15
    # Step 1: log returns
    log_ret = np.diff(np.log(np.concatenate([[close[0]], close])))
    # Step 2: EMA1
    ema1 = pd.Series(log_ret).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 3: EMA2
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 4: EMA3
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix_raw = 100 * np.diff(np.concatenate([[0], ema3])) / np.concatenate([[1e-10], ema3[:-1]])
    
    # Weekly trend filter: EMA50 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema50_1w
    downtrend_1w = df_1w['close'].values < ema50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after TRIX warmup
        trix_now = trix_raw[i]
        trix_prev = trix_raw[i-1]
        vol_spike = volume_spike[i]
        uptrend_weekly = uptrend_1w_aligned[i]
        downtrend_weekly = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: TRIX crosses above zero with volume spike and weekly uptrend
            if trix_prev <= 0 and trix_now > 0 and vol_spike and uptrend_weekly:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume spike and weekly downtrend
            elif trix_prev >= 0 and trix_now < 0 and vol_spike and downtrend_weekly:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses back below zero or weekly trend turns down
            if trix_now < 0 or not uptrend_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses back above zero or weekly trend turns up
            if trix_now > 0 or not downtrend_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals