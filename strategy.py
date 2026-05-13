#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) on the daily chart act as strong support/resistance. 
Breakout above R3 or below S3 with volume confirmation and daily trend filter (EMA34) captures 
strong momentum moves. Works in both bull and bear markets by trading breakouts in the direction 
of the higher timeframe trend. Target: 20-50 trades/year per symbol.
"""

name = "4h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C, H, L are from previous daily close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema_34_1d
    downtrend_1d = df_1d['close'].values < ema_34_1d
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if position == 0:
            # LONG: break above R3, volume confirmation, daily uptrend
            if close[i] > R3_aligned[i] and vol_ok[i] and uptrend_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S3, volume confirmation, daily downtrend
            elif close[i] < S3_aligned[i] and vol_ok[i] and downtrend_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price re-enters below R3 or trend change
            if close[i] < R3_aligned[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price re-enters above S3 or trend change
            if close[i] > S3_aligned[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals