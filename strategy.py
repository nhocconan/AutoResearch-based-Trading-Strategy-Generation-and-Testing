#!/usr/bin/env python3
# 4H_KAMA_TRIX_Confluence_v1
# Hypothesis: Combines KAMA trend direction with TRIX momentum and volume confirmation on 4h timeframe.
# KAMA adapts to market noise, reducing false signals in ranging markets. TRIX captures momentum shifts.
# Volume spike confirms institutional participation. Designed for low trade frequency (<30/year) to avoid fee drag.
# Works in bull markets via KAMA uptrend + TRIX cross-up, and in bear markets via KAMA downtrend + TRIX cross-down.
# Uses 1d timeframe for trend context to avoid whipsaws.

name = "4H_KAMA_TRIX_Confluence_v1"
timeframe = "4h"
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
    
    # Get 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    # Calculate ER (Efficiency Ratio) and SSC (Smoothing Constant)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # TRIX on 1d close
    close_1d = df_1d['close'].values
    # Triple EMA
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    # TRIX = 100 * (ema3 - ema3_prev) / ema3_prev
    ema3_values = ema3.values
    trix = np.zeros_like(ema3_values)
    trix[1:] = 100 * (ema3_values[1:] - ema3_values[:-1]) / ema3_values[:-1]
    trix = np.where(np.isnan(trix), 0, trix)
    
    # Align KAMA and TRIX to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(trix_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA + TRIX turning up + volume spike
            if (close[i] > kama_aligned[i] and 
                trix_aligned[i] > trix_aligned[i-1] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + TRIX turning down + volume spike
            elif (close[i] < kama_aligned[i] and 
                  trix_aligned[i] < trix_aligned[i-1] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or TRIX turns down
            if close[i] < kama_aligned[i] or trix_aligned[i] < trix_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or TRIX turns up
            if close[i] > kama_aligned[i] or trix_aligned[i] > trix_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals