#!/usr/bin/env python3
"""
4h_KAMA_Trend_Volume_Regime_v1
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 4h to capture adaptive trend direction, combined with volume confirmation and a 12h chop regime filter to avoid whipsaws. Designed to work in both bull and bear markets by adapting to market conditions - KAMA reduces lag in trends and avoids false signals in ranges, while volume confirms breakouts and chop filter ensures we only trade in favorable regimes. Target: 20-50 trades/year per symbol.
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
    
    # Calculate KAMA on 4h close
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_values = kama(close, period=10, fast=2, slow=30)
    
    # 12h chop regime filter (Choppiness Index)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) sum
    atr_sum = np.full(len(close_12h), np.nan)
    for i in range(14, len(close_12h)):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Sum of high-low ranges
    hl_sum = np.full(len(close_12h), np.nan)
    for i in range(14, len(close_12h)):
        hl_sum[i] = np.sum(high_12h[i-13:i+1] - low_12h[i-13:i+1])
    
    # Choppiness Index
    chop = np.full(len(close_12h), np.nan)
    for i in range(14, len(close_12h)):
        if hl_sum[i] != 0:
            chop[i] = 100 * np.log10(atr_sum[i] / hl_sum[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure KAMA and volume MA ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_values[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA with volume spike and chop < 61.8 (trending)
            if (close[i] > kama_values[i] and vol_spike[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA with volume spike and chop < 61.8 (trending)
            elif (close[i] < kama_values[i] and vol_spike[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA or chop > 61.8 (range)
            if (close[i] < kama_values[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA or chop > 61.8 (range)
            if (close[i] > kama_values[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0