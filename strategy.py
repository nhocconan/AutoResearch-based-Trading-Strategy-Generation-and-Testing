#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA with RSI filter and chop regime
# Long: KAMA slope > 0 + RSI(14) > 50 + Chop(14) < 61.8 (trending regime)
# Short: KAMA slope < 0 + RSI(14) < 50 + Chop(14) < 61.8 (trending regime)
# Uses 1d ATR for chop calculation and 1d close for KAMA efficiency ratio
# Chop filters out sideways markets, KAMA adapts to volatility, RSI confirms momentum
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear by only trading in trending regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for chop and KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Chop calculation on 1d
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(high_1d[i] - low_1d[i], 
                           abs(high_1d[i] - close_1d[i-1]), 
                           abs(low_1d[i] - close_1d[i-1]))
        if i < 14:
            atr_1d[i] = np.nan
        else:
            if i == 14:
                atr_1d[i] = np.mean(tr_1d[0:15])
            else:
                atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Chop = 100 * log15(sum(atr14) / (max(high14) - min(low14)))
    chop = np.full(len(close_1d), np.nan)
    for i in range(27, len(close_1d)):  # 14+13 for ATR + 14 for HHV/LLV
        sum_atr = np.sum(atr_1d[i-13:i+1])
        hh = np.max(high_1d[i-13:i+1])
        ll = np.min(low_1d[i-13:i+1])
        if hh > ll:
            chop[i] = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(15)
        else:
            chop[i] = 50.0
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # KAMA on 1d
    # Efficiency Ratio
    change_1d = np.abs(np.diff(close_1d, k=10))  # 10-period change
    abs_change_1d = np.sum(np.abs(np.diff(close_1d)), axis=1)  # needs correction
    
    # Recalculate ER properly
    er = np.full(len(close_1d), np.nan)
    for i in range(10, len(close_1d)):
        direction = abs(close_1d[i] - close_1d[i-10])
        volatility = 0
        for j in range(i-9, i+1):
            volatility += abs(close_1d[j] - close_1d[j-1])
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(er[i]):
            fast_sc = 2 / (2 + 1)   # EMA(2)
            slow_sc = 2 / (30 + 1)  # EMA(30)
            sc[i] = (er[i] * fast_sc + (1 - er[i]) * slow_sc) ** 2
        else:
            sc[i] = np.nan
    
    # KAMA calculation
    kama = np.full(len(close_1d), np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI on 12h
    rsi = np.full(n, np.nan)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        change = close[i] - close[i-1]
        if change > 0:
            gains[i] = change
            losses[i] = 0
        else:
            gains[i] = 0
            losses[i] = -change
    
    # Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gains[1:15])
            avg_loss[i] = np.mean(losses[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gains[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + losses[i]) / 14
    
    for i in range(14, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        elif avg_gain[i] == 0:
            rsi[i] = 0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # KAMA slope
    kama_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(kama_aligned[i]) and not np.isnan(kama_aligned[i-1]):
            kama_slope[i] = kama_aligned[i] - kama_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + trending regime (chop < 61.8)
            if (kama_slope[i] > 0 and 
                rsi[i] > 50 and
                chop_val < 61.8):
                position = 1
                signals[i] = position_size
            # Short: KAMA down + RSI < 50 + trending regime (chop < 61.8)
            elif (kama_slope[i] < 0 and 
                  rsi[i] < 50 and
                  chop_val < 61.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA down or chop > 61.8 (choppy)
            if (kama_slope[i] < 0 or
                chop_val >= 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA up or chop > 61.8 (choppy)
            if (kama_slope[i] > 0 or
                chop_val >= 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0