#!/usr/bin/env python3
"""
4h KAMA Trend + RSI Filter + Chop Regime Filter.
Long when KAMA is rising, RSI > 50, and Chop < 61.8 (trending market).
Short when KAMA is falling, RSI < 50, and Chop < 61.8 (trending market).
Exit when opposite signal or Chop > 61.8 (range market).
Designed for low frequency (20-50 trades/year) to minimize fee drag.
Uses KAMA for trend, RSI for momentum filter, and Chop for regime filter.
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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
    # Proper ER calculation
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(er_len, n):
        if i >= er_len:
            price_change = np.abs(close[i] - close[i-er_len])
            abs_changes = np.sum(np.abs(np.diff(close[i-er_len:i+1])))
            if abs_changes > 0:
                er[i] = price_change / abs_changes
            else:
                er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[14:15])
            avg_loss[i] = np.mean(loss[14:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.zeros(n)
    rsi[:] = 50
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Choppy Index (CHOP) - using 14-period
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.zeros(n)
    atr[:] = np.nan
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate highest high and lowest low over ATR period
    hh = np.zeros(n)
    ll = np.zeros(n)
    hh[:] = np.nan
    ll[:] = np.nan
    for i in range(atr_period-1, n):
        hh[i] = np.max(high[i-atr_period+1:i+1])
        ll[i] = np.min(low[i-atr_period+1:i+1])
    
    # Chop calculation
    chop = np.zeros(n)
    chop[:] = 50
    for i in range(atr_period, n):
        if atr[i] > 0 and hh[i] > ll[i]:
            sum_tr = np.sum(tr[i-atr_period+1:i+1])
            chop[i] = 100 * np.log10(sum_tr / (atr[i] * atr_period)) / np.log10(atr_period)
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14)
    start_idx = max(er_len + 1, 14, atr_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_now = rsi[i]
        chop_now = chop[i]
        
        # Regime filter: Chop < 61.8 = trending market
        trending_regime = chop_now < 61.8
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + trending regime
            if kama_now > kama_prev and rsi_now > 50 and trending_regime:
                signals[i] = size
                position = 1
            # Short: KAMA falling + RSI < 50 + trending regime
            elif kama_now < kama_prev and rsi_now < 50 and trending_regime:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 50 OR Chop > 61.8 (range)
            if kama_now < kama_prev or rsi_now < 50 or chop_now > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 50 OR Chop > 61.8 (range)
            if kama_now > kama_prev or rsi_now > 50 or chop_now > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_RSI_Chop_Trend"
timeframe = "4h"
leverage = 1.0