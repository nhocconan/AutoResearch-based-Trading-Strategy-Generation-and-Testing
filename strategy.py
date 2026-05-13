#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeConfirmation
Hypothesis: Price breaking above Camarilla R1 or below S1 on 12h with 1d trend alignment and volume confirmation provides high-probability trend-following entries. Works in bull/bear markets by following the 1d trend direction. Target: 12-37 trades/year per symbol.
"""

name = "12h_1d_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r1 = close + range_hl * 1.1 / 12
    s1 = close - range_hl * 1.1 / 12
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # 1d volume average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period 1d MA
        vol_confirm = volume[i] > 1.5 * vol_ma if not np.isnan(vol_ma) else False
        
        if position == 0:
            # LONG: Price breaks above R1, 1d uptrend, volume confirmation
            if close[i] > r1[i] and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1, 1d downtrend, volume confirmation
            elif close[i] < s1[i] and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or 1d trend turns down
            if close[i] < s1[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or 1d trend turns up
            if close[i] > r1[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals