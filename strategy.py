#!/usr/bin/env python3
# 6h_Pullback_to_EMA_With_Volume_and_Trend
# Hypothesis: In strong trends (determined by weekly EMA34), price pulls back to the daily EMA89, offering high-probability continuation entries.
# A volume spike confirms institutional participation in the bounce/break. This strategy avoids chasing breakouts and instead enters on retracements.
# Works in both bull and bear markets by aligning with the weekly trend direction.
# Target: 20-40 trades per year by requiring weekly trend alignment, EMA pullback, and volume confirmation.

name = "6h_Pullback_to_EMA_With_Volume_and_Trend"
timeframe = "6h"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for EMA89 (pullback target)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:
        return np.zeros(n)
    
    # Weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Daily EMA89 for pullback entries
    ema_89_1d = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(ema_89_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly uptrend (price > weekly EMA34) + pullback to daily EMA89 + volume spike
            if (close[i] > ema_34_1w_aligned[i] and 
                close[i] <= ema_89_1d_aligned[i] * 1.005 and  # Allow small overshoot
                close[i] >= ema_89_1d_aligned[i] * 0.995 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend (price < weekly EMA34) + pullback to daily EMA89 + volume spike
            elif (close[i] < ema_34_1w_aligned[i] and 
                  close[i] >= ema_89_1d_aligned[i] * 0.995 and  # Allow small overshoot
                  close[i] <= ema_89_1d_aligned[i] * 1.005 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below daily EMA89 or weekly trend changes
            if close[i] < ema_89_1d_aligned[i] * 0.995 or \
               close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above daily EMA89 or weekly trend changes
            if close[i] > ema_89_1d_aligned[i] * 1.005 or \
               close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals