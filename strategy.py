#!/usr/bin/env python3
name = "12h_KAMA_Trend_RSI_Chop"
timeframe = "12h"
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
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on close
    def kama(price, period=10):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
        kama = np.zeros_like(price)
        kama[:period] = price[:period]
        for i in range(period, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10)
    kama_dir = np.zeros(n, dtype=int)
    kama_dir[1:] = np.where(kama_val[1:] > kama_val[:-1], 1, -1)
    
    # RSI(14)
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
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_val = rsi(close, 14)
    
    # Chopiness Index (14) on 1d
    def chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            if np.isnan(tr[i]):
                atr[i] = atr[i-1]
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period if i >= period else np.mean(tr[1:i+1])
        sum_atr = np.nansum(atr.reshape(-1, period), axis=1) * period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(len(close)):
            start = max(0, i-period+1)
            hh[i] = np.max(high[start:i+1])
            ll[i] = np.min(low[start:i+1])
        chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(period)
        return chop
    
    chop_val = chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_val)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_val[i]) or 
            np.isnan(rsi_val[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: range when > 61.8
        in_range = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA up + RSI > 50 in range
            if kama_dir[i] == 1 and rsi_val[i] > 50 and in_range:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 in range
            elif kama_dir[i] == -1 and rsi_val[i] < 50 and in_range:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down or RSI < 40
            if kama_dir[i] == -1 or rsi_val[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up or RSI > 60
            if kama_dir[i] == 1 or rsi_val[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend with RSI filter in choppy markets (Chop > 61.8) on 12h.
# Long when KAMA rising and RSI > 50 in range-bound conditions.
# Short when KAMA falling and RSI < 50 in range-bound conditions.
# Uses 1d Chop filter to identify ranging markets where mean reversion works.
# Works in both bull and bear markets by adapting to range conditions.
# Target: 50-150 total trades over 4 years (12-37/year) as per experiment guidelines.