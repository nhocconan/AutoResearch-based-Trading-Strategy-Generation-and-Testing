#!/usr/bin/env python3
"""
1d_1w_kama_rsi_volatility_breakout
Hypothesis: Daily KAMA trend with RSI momentum and volatility breakout.
Uses 1-week KAMA for trend filter and 1-day RSI + volatility expansion for entries.
Works in bull/bear by trading with weekly trend and entering on volatility spikes.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

name = "1d_1w_kama_rsi_volatility_breakout"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w KAMA for trend direction
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_period] = close[er_period]
        for i in range(er_period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1w = kama(close_1w, er_period=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # 1d RSI(14) for momentum
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = rsi(close, period=14)
    
    # Volatility expansion: ATR(5) > 1.5 * ATR(20)
    def atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr5 = atr(high, low, close, period=5)
    atr20 = atr(high, low, close, period=20)
    vol_expansion = atr5 > (atr20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(atr5[i]) or np.isnan(atr20[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > KAMA (uptrend) AND RSI > 55 AND volatility expansion
        if (close[i] > kama_1w_aligned[i] and rsi_1d[i] > 55 and vol_expansion[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < KAMA (downtrend) AND RSI < 45 AND volatility expansion
        elif (close[i] < kama_1w_aligned[i] and rsi_1d[i] < 45 and vol_expansion[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or RSI crosses back to neutral
        elif position == 1 and rsi_1d[i] < 45:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi_1d[i] > 55:
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