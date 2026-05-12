#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filtered_By_Volume_and_Camarilla
Hypothesis: KAMA adapts to trend strength, so using it as a trend filter reduces whipsaws in both bull and bear markets.
Combined with Camarilla R1/S1 breakout and volume confirmation for entry, and KAMA reversal for exit.
Only enters when price breaks Camarilla level in direction of KAMA trend, with volume spike.
Designed to be selective: ~25-35 trades/year per symbol to avoid fee drag.
"""

name = "4h_KAMA_Trend_Filtered_By_Volume_and_Camarilla"
timeframe = "4h"
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
    
    # Volume spike: >1.6x 40-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    volume_spike = volume > (1.6 * vol_ma)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # 4h data for KAMA trend filter
    # KAMA parameters: ER decay = 2/(2+1) = 0.67, SC = [ER*(fastest-slowest)+slowest]^2
    # Using ER=10, fastest=2, slowest=30 as common settings
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)  # Avoid division by zero
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2  # fastest=2 -> 2/(2+1)=0.6667, slowest=30 -> 2/(30+1)=0.0645
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align all indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)  # Self-align for 4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(kama_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + KAMA uptrend (price > KAMA) + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > kama_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + KAMA downtrend (price < KAMA) + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < kama_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA (trend change)
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA (trend change)
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals