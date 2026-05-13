#!/usr/bin/env python3
"""
1d_1w_1wKeltner_MeanReversion
Hypothesis: In high-volatility regimes (1w ATR expansion), price reverts to the 1w KAMA.
Enter long when price touches lower Keltner channel (KAMA - 1.5*ATR) in an oversold RSI condition.
Enter short when price touches upper Keltner channel (KAMA + 1.5*ATR) in an overbought RSI condition.
Use 1w trend filter to avoid counter-trend trades. Designed for low-frequency, high-conviction trades.
Target: 7-25 trades/year per symbol.
"""

name = "1d_1w_1wKeltner_MeanReversion"
timeframe = "1d"
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
    
    # Get 1w data for KAMA, ATR, RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w KAMA (using close prices)
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        abs_change = np.abs(np.diff(close, prepend=close[0]))
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if np.sum(abs_change[i-length+1:i+1]) > 0:
                er[i] = change[i] / np.sum(abs_change[i-length+1:i+1])
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1w = calculate_kama(close_1w, length=10, fast=2, slow=30)
    
    # Calculate 1w ATR
    def calculate_atr(high, low, close, length=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = np.zeros_like(close)
        for i in range(length, len(close)):
            if i == length:
                atr[i] = np.nanmean(tr[i-length+1:i+1])
            else:
                atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        return atr
    
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, length=14)
    
    # Calculate 1w RSI
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        for i in range(length, len(close)):
            if i == length:
                avg_gain[i] = np.mean(gain[i-length+1:i+1])
                avg_loss[i] = np.mean(loss[i-length+1:i+1])
            else:
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1w = calculate_rsi(close_1w, length=14)
    
    # 1w trend: 50 EMA
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align all 1w indicators to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Calculate Keltner channels
    upper_keltner = kama_1w_aligned + 1.5 * atr_1w_aligned
    lower_keltner = kama_1w_aligned - 1.5 * atr_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        kama = kama_1w_aligned[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        rsi = rsi_1w_aligned[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: price touches lower Keltner, RSI oversold, in 1w uptrend (avoid counter-trend)
            if close[i] <= lower and rsi < 30 and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: price touches upper Keltner, RSI overbought, in 1w downtrend
            elif close[i] >= upper and rsi > 70 and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses above KAMA or RSI overbought
            if close[i] >= kama or rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below KAMA or RSI oversold
            if close[i] <= kama or rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals