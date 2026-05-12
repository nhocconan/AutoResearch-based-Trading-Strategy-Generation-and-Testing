#!/usr/bin/env python3
"""
12h_KAMA_Reversal_With_Volume_Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing timely trend reversals. Combined with volume spikes and a 1-day ADX trend filter, it captures mean-reversion moves in ranging markets and avoids whipsaws in strong trends. Target: 15-25 trades/year per symbol.
"""

name = "12h_KAMA_Reversal_With_Volume_Confirmation"
timeframe = "12h"
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
    
    # Volume spike: >1.6x 30-period average (to reduce frequency)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.6 * vol_ma)
    
    # KAMA calculation (ER = Efficiency Ratio, SC = Smoothing Constant)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), 
                               np.abs(low_1d[1:] - low_1d[:-1])))
    # Pad to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # same timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after KAMA warmup
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above KAMA + low ADX (ranging market) + volume spike
            if (close[i] > kama_aligned[i] and 
                close[i-1] <= kama_aligned[i-1] and 
                adx_aligned[i] < 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA + low ADX + volume spike
            elif (close[i] < kama_aligned[i] and 
                  close[i-1] >= kama_aligned[i-1] and 
                  adx_aligned[i] < 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR ADX rises (trend developing)
            if (close[i] < kama_aligned[i]) or (adx_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR ADX rises
            if (close[i] > kama_aligned[i]) or (adx_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals