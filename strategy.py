#!/usr/bin/env python3
"""
4h_Supertrend_10_3_Plus_Volume_And_Chop_Filter
Hypothesis: Supertrend (ATR=10, multiplier=3) identifies the trend direction. 
Only take trades when volume is above average (confirming momentum) and the market is not too choppy (Choppiness Index > 61.8 indicates ranging, so we avoid).
This combination should work in both bull and bear markets by following strong trends with confirmation, avoiding whipsaws in sideways markets.
Target: ~20-40 trades/year, position size 0.25.
"""

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
    
    # Calculate ATR(10)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(10, n):
        atr[i] = np.mean(tr[i-9:i+1])
    
    # Calculate Supertrend
    hl2 = (high + low) / 2
    upperband = hl2 + 3 * atr
    lowerband = hl2 - 3 * atr
    
    supertrend = np.full(n, np.nan)
    uptrend = np.full(n, True)
    
    for i in range(1, n):
        if np.isnan(atr[i-1]) or np.isnan(close[i-1]):
            continue
        if close[i] > upperband[i-1]:
            uptrend[i] = True
        elif close[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        supertrend[i] = lowerband[i] if uptrend[i] else upperband[i]
    
    # Get 1D data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr1 = high_1d[i] - low_1d[i]
        tr2 = np.abs(high_1d[i] - close_1d[i-1])
        tr3 = np.abs(low_1d[i] - close_1d[i-1])
        tr = max(tr1, tr2, tr3)
        if i == 1:
            atr_1d[i] = tr
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_atr = np.sum(atr_1d[i-13:i+1])
        hh = np.max(high_1d[i-13:i+1])
        ll = np.min(low_1d[i-13:i+1])
        if hh - ll != 0:
            chop[i] = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(14)
    
    # Align Choppiness to 4h
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14)  # need volume MA and chop
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend[i]) or np.isnan(chop_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Chop filter: avoid ranging markets (Choppiness > 61.8)
        not_choppy = chop_4h[i] <= 61.8
        
        if position == 0:
            # Long: uptrend + volume + not choppy
            if uptrend[i] and vol_confirmed and not_choppy:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + not choppy
            elif not uptrend[i] and vol_confirmed and not_choppy:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: trend changes
            if not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend changes
            if uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Supertrend_10_3_Plus_Volume_And_Chop_Filter"
timeframe = "4h"
leverage = 1.0