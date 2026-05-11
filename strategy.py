#!/usr/bin/env python3
name = "12h_KAMA_Trend_With_Volume_And_Chop"
timeframe = "12h"
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
    
    # 1d data for trend filter and KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum of |diff| over 10 periods
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[29] = close_1d[29]  # start after enough data
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 12h KAMA for entry timing
    # Calculate KAMA on 12h data
    change_12h = np.abs(np.diff(close, n=10))
    volatility_12h = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    volatility_12h = np.where(volatility_12h == 0, 1e-10, volatility_12h)
    er_12h = change_12h / volatility_12h
    sc_12h = (er_12h * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_12h = np.full_like(close, np.nan, dtype=np.float64)
    kama_12h[29] = close[29]
    for i in range(30, len(close)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close[i] - kama_12h[i-1])
    
    # Choppiness Index on 1d (14-period)
    def choppiness_index(high, low, close, period=14):
        atr = []
        for i in range(len(close)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        atr = np.array(atr)
        # Sum of ATR over period
        sum_atr = np.convolve(atr, np.ones(period), 'valid')
        # Highest high and lowest low over period
        max_h = np.zeros_like(close)
        min_l = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                max_h[i] = np.max(high[:i+1])
                min_l[i] = np.min(low[:i+1])
            else:
                max_h[i] = np.max(high[i-period+1:i+1])
                min_l[i] = np.min(low[i-period+1:i+1])
        range_maxmin = max_h - min_l
        # Avoid division by zero
        range_maxmin = np.where(range_maxmin == 0, 1e-10, range_maxmin)
        chop = 100 * np.log10(sum_atr / range_maxmin) / np.log10(period)
        # Pad to same length
        chop_full = np.full_like(close, np.nan)
        chop_full[period-1:] = chop
        return chop_full
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 35  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(kama_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA trending up (price > KAMA), above EMA34, and chop < 61.8 (trending)
            if (close[i] > kama_12h[i] and
                close[i] > ema34_1d_aligned[i] and
                chop_1d_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down (price < KAMA), below EMA34, and chop < 61.8
            elif (close[i] < kama_12h[i] and
                  close[i] < ema34_1d_aligned[i] and
                  chop_1d_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or chop > 61.8 (choppy)
            if (close[i] < kama_12h[i] or
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA or chop > 61.8
            if (close[i] > kama_12h[i] or
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals