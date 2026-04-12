#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_kama_rsi_chop
# Uses Kaufman's Adaptive Moving Average (KAMA) on 1d timeframe for trend direction.
# Enters long when KAMA slope > 0 and RSI(14) > 50 on 4h, short when slope < 0 and RSI < 50.
# Uses Choppiness Index (14) on 4h to avoid chop: only trade when CHOP < 40 (trending).
# Exits when RSI crosses back to 50 or KAMA slope changes sign.
# Designed for low trade frequency (~20-40 trades/year) to minimize fee drag.
# Works in trending markets via KAMA/RSI alignment and avoids whipsaw via CHOP filter.

name = "4h_1d_kama_rsi_chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 9:
            er[i] = np.nan
        else:
            direction = np.abs(close_1d[i] - close_1d[i-9])
            volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility_sum if volatility_sum != 0 else 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_slope = np.diff(kama, prepend=0)
    
    # Align daily KAMA slope to 4h timeframe
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope)
    
    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on 4h
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(tr, axis=0) / (max_hh - min_ll)) / np.log10(14)  # placeholder, fix
    # Proper CHOP calculation
    chop = np.zeros(n)
    for i in range(14, n):
        atr_sum = np.sum(tr[i-13:i+1])
        hh_ll = max_hh[i] - min_ll[i]
        chop[i] = 100 * np.log10(atr_sum / hh_ll) / np.log10(14) if hh_ll != 0 else 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if np.isnan(kama_slope_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Only trade in trending market (CHOP < 40)
        if chop[i] >= 40:
            # Hold current position if choppy
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: KAMA up and RSI > 50
        if kama_slope_aligned[i] > 0 and rsi[i] > 50 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: KAMA down and RSI < 50
        elif kama_slope_aligned[i] < 0 and rsi[i] < 50 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: RSI crosses 50 or KAMA slope changes sign
        elif position == 1 and (rsi[i] <= 50 or kama_slope_aligned[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] >= 50 or kama_slope_aligned[i] >= 0):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals