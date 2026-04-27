#!/usr/bin/env python3
"""
12h KAMA + RSI + Chop Filter with 1d Trend Filter.
Long when: 1) KAMA rising (bullish momentum), 2) RSI < 30 (oversold), 3) Chop > 61.8 (range), 4) Price > 1d EMA50 (bullish trend).
Short when: 1) KAMA falling (bearish momentum), 2) RSI > 70 (overbought), 3) Chop > 61.8 (range), 4) Price < 1d EMA50 (bearish trend).
Exit when momentum reverses or trend changes.
Designed for 12h timeframe: targets 50-150 total trades over 4 years (12-37/year).
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
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    for i in range(1, n):
        if volatility[i-1] > 0:
            er[i] = change[i] / volatility[i-1]
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop(14)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        if i < 14:
            highest_high[i] = np.max(high[:i+1])
            lowest_low[i] = np.min(low[:i+1])
        else:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])
    chop = np.where((highest_high - lowest_low) > 0, 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14), 50)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA (stable), RSI (14), Chop (14), 1d EMA (50)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_now = rsi[i]
        chop_now = chop[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # KAMA direction: rising/falling
        kama_rising = kama_now > kama_prev
        kama_falling = kama_now < kama_prev
        
        if position == 0:
            # Long: KAMA rising + RSI < 30 + Chop > 61.8 + bullish trend
            if kama_rising and rsi_now < 30 and chop_now > 61.8 and price > ema_trend:
                signals[i] = size
                position = 1
            # Short: KAMA falling + RSI > 70 + Chop > 61.8 + bearish trend
            elif kama_falling and rsi_now > 70 and chop_now > 61.8 and price < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA falls or trend turns bearish
            if not kama_rising or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA rises or trend turns bullish
            if not kama_falling or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_RSI_Chop_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0