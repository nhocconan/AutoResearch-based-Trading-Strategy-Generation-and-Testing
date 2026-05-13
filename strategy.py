# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla pivot levels on 1d provide robust support/resistance zones. 
Breakout above R3 or below S3 with 1d EMA34 trend filter and volume spike (1.5x average) 
captures strong momentum moves. Works in both bull and bear markets by following 
the higher timeframe trend. Target: 20-40 trades/year per symbol.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
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
    
    # Calculate Camarilla levels using previous 12h bar's range
    # Shift high/low/close by 1 to use previous bar for pivot calculation
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = high[0]  # fill first value
    plow[0] = low[0]
    pclose[0] = close[0]
    
    # Camarilla calculations
    range_val = phigh - plow
    R3 = pclose + range_val * 1.1 / 4
    S3 = pclose - range_val * 1.1 / 4
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema_34_1d
    downtrend_1d = df_1d['close'].values < ema_34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after volume MA warmup
        if position == 0:
            # LONG: close > R3, volume spike, 1d uptrend
            if close[i] > R3[i] and vol_spike[i] and uptrend_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: close < S3, volume spike, 1d downtrend
            elif close[i] < S3[i] and vol_spike[i] and downtrend_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < S3 (reversal to opposite level)
            if close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > R3 (reversal to opposite level)
            if close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals