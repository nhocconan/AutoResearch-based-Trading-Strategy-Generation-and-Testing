#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_ChopFilter_V1
KAMA direction + RSI + chop filter on 12h timeframe.
Uses KAMA for adaptive trend, RSI for momentum, and Choppiness Index for regime filter.
Works in both bull and bear markets by avoiding ranging conditions.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d KAMA (10-period ER, 2/30 fast/slow) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 9:  # 10-period lookback
            net_change = np.abs(close_1d[i] - close_1d[i-9])
            total_change = np.sum(volatility[i-9:i+1])
            er[i] = net_change / total_change if total_change != 0 else 0
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 1d RSI(14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Wilder's smoothing
    for i in range(len(close_1d)):
        if i < 14:
            if i > 0:
                avg_gain[i] = np.sum(gain[1:i+1]) / 14
                avg_loss[i] = np.sum(loss[1:i+1]) / 14
            else:
                avg_gain[i] = 0
                avg_loss[i] = 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Choppiness Index (14-period) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close_1d[0]), np.abs(low[0] - close_1d[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i >= 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Highest high and lowest low over 14 periods
    hh = np.zeros_like(close_1d)
    ll = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 13:
            hh[i] = np.max(high[i-13:i+1])
            ll[i] = np.min(low[i-13:i+1])
        else:
            hh[i] = np.max(high[:i+1])
            ll[i] = np.min(low[:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 13 and atr[i] > 0:
            sum_atr = np.sum(atr[i-13:i+1])
            chop[i] = 100 * np.log10(sum_atr / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # === Align indicators to 12h timeframe ===
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Volume confirmation ===
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    warmup = 100
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry when flat
        if position == 0:
            # Long: price > KAMA, RSI > 50, chop < 61.8 (trending)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < KAMA, RSI < 50, chop < 61.8 (trending)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit long
        elif position == 1:
            if (close[i] <= kama_aligned[i] or 
                rsi_aligned[i] < 40 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        # Exit short
        elif position == -1:
            if (close[i] >= kama_aligned[i] or 
                rsi_aligned[i] > 60 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RSI_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0