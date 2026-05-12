#!/usr/bin/env python3
"""
4H_KAMA_ADX_VOLUME_FILTER
Hypothesis: KAMA trend direction combined with ADX trend strength and volume confirmation.
KAMA adapts to market noise, reducing false signals in ranging markets. ADX filters for
trending conditions (ADX > 25), and volume ensures institutional participation.
Works in bull markets by capturing trends and in bear markets by avoiding false signals
during low-volatility periods. Targets ~25-35 trades/year on 4h to minimize fee drag.
"""
name = "4H_KAMA_ADX_VOLUME_FILTER"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, FAST = 2, SLOW = 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(10, n):
        er[i] = np.abs(close[i] - close[i-10]) / np.sum(volatility[i-9:i+1]) if np.sum(volatility[i-9:i+1]) > 0 else 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ADX calculation (period = 14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        plus_dm[i] = high[i] - high[i-1] if high[i] - high[i-1] > high[i-1] - low[i] and high[i] - high[i-1] > 0 else 0
        minus_dm[i] = high[i-1] - low[i] if high[i-1] - low[i] > high[i] - high[i-1] and high[i-1] - low[i] > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    for i in range(14, n):
        plus_di[i] = 100 * (np.sum(plus_dm[i-13:i+1]) / np.sum(tr[i-13:i+1])) if np.sum(tr[i-13:i+1]) > 0 else 0
        minus_di[i] = 100 * (np.sum(minus_dm[i-13:i+1]) / np.sum(tr[i-13:i+1])) if np.sum(tr[i-13:i+1]) > 0 else 0
        dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) > 0 else 0
    
    adx[0] = 0
    for i in range(1, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14 if i >= 14 else 0
    
    # Volume confirmation (20-period average)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros(len(close_1d))
    for i in range(50, len(close_1d)):
        ema_50_1d[i] = np.mean(close_1d[i-49:i+1]) if i >= 50 else close_1d[i]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA, ADX > 25, volume above average, and above 1d EMA50
            if (close[i] > kama[i] and 
                adx[i] > 25 and 
                volume[i] > vol_ma[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, ADX > 25, volume above average, and below 1d EMA50
            elif (close[i] < kama[i] and 
                  adx[i] > 25 and 
                  volume[i] > vol_ma[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals