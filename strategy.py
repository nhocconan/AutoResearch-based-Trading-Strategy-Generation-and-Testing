#!/usr/bin/env python3
name = "12h_Trix_Volume_Spike_1dTrend"
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
    
    # TRIX: 1-period rate of change of triple EMA(15)
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)
    for i in range(n):
        if i == 0:
            ema1[i] = close[i]
            ema2[i] = close[i]
            ema3[i] = close[i]
        else:
            ema1[i] = 0.125 * close[i] + 0.875 * ema1[i-1]  # 2/(15+1) = 0.125
            ema2[i] = 0.125 * ema1[i] + 0.875 * ema2[i-1]
            ema3[i] = 0.125 * ema2[i] + 0.875 * ema3[i-1]
    trix = np.full(n, np.nan)
    for i in range(1, n):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # TRIX signal line: EMA(9) of TRIX
    trix_signal = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            trix_signal[i] = trix[i] if not np.isnan(trix[i]) else 0
        else:
            if np.isnan(trix[i]):
                trix_signal[i] = trix_signal[i-1]
            else:
                trix_signal[i] = 0.2 * trix[i] + 0.8 * trix_signal[i-1]  # 2/(9+1) = 0.2
    
    # 1d trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        trix_bullish = trix[i] > trix_signal[i]
        trix_bearish = trix[i] < trix_signal[i]
        
        if position == 0:
            # LONG: TRIX bullish crossover + 1d uptrend + volume spike
            if trix_bullish and close[i] > ema34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX bearish crossover + 1d downtrend + volume spike
            elif trix_bearish and close[i] < ema34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX bearish crossover or trend breaks
            if trix_bearish or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX bullish crossover or trend breaks
            if trix_bullish or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals