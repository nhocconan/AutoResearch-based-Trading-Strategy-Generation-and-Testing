#!/usr/bin/env python3
"""
4h_KAMA_Direction_Plus_RSI_And_Chop
Hypothesis: 4-hour KAMA direction filter (trend identification) combined with RSI extremes and 1-day Choppiness index regime filter.
KAMA identifies the trend direction (bull/bear), RSI identifies entry points within that trend, and Choppiness identifies whether we are in a trending or ranging market.
This combination should work in both bull and bear markets by adapting to the prevailing trend while avoiding whipsaws in ranging conditions.
Target: 20-50 trades per year.
"""

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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend direction
    # Using 10-period ER and slow/fast EMA periods as standard
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This needs fixing
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if np.sum(np.abs(np.diff(close[i-9:i+1]))) > 0:
            er[i] = np.abs(close[i] - close[i-9]) / np.sum(np.abs(np.diff(close[i-9:i+1])))
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA 2
    slow_sc = 2 / (30 + 1)  # EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1-day Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14)
    atr_period = 14
    atr_1d = np.zeros(len(high_1d))
    if len(high_1d) >= atr_period:
        atr_1d[atr_period-1] = np.mean(tr1[:atr_period])
        for i in range(atr_period, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period-1) + tr1[i]) / atr_period
    
    # Sum of ATR over period
    sum_atr_1d = np.zeros(len(high_1d))
    for i in range(atr_period-1, len(high_1d)):
        sum_atr_1d[i] = np.sum(atr_1d[i-atr_period+1:i+1])
    
    # Choppiness Index
    chop = np.zeros(len(high_1d))
    for i in range(atr_period-1, len(high_1d)):
        if sum_atr_1d[i] > 0:
            max_high = np.max(high_1d[i-atr_period+1:i+1])
            min_low = np.min(low_1d[i-atr_period+1:i+1])
            chop[i] = 100 * np.log10(sum_atr_1d[i] / (max_high - min_low)) / np.log10(atr_period)
        else:
            chop[i] = 50  # Neutral
    
    # Align 1-day Choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi_period = 14
    
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[1:rsi_period+1])
            avg_loss[i] = np.mean(loss[1:rsi_period+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # KAMA, volume, RSI
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Market regime: Chop > 50 = ranging, Chop < 50 = trending
        is_trending = chop_aligned[i] < 50
        
        if position == 0:
            # Long conditions: KAMA up (bullish trend), RSI oversold, volume spike, in trending market
            if (close[i] > kama[i] and rsi[i] < 30 and vol_spike[i] and is_trending):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down (bearish trend), RSI overbought, volume spike, in trending market
            elif (close[i] < kama[i] and rsi[i] > 70 and vol_spike[i] and is_trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI overbought
            if (close[i] < kama[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI oversold
            if (close[i] > kama[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_Plus_RSI_And_Chop"
timeframe = "4h"
leverage = 1.0