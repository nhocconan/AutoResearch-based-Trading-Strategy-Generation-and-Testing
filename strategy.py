#!/usr/bin/env python3
# 4h_KAMA_Trend_Filter_v3
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a dynamic trend filter.
# Combined with volume confirmation and a volatility regime filter (ATR-based), it aims to capture strong trends
# while avoiding whipsaws in choppy markets. Uses daily timeframe for trend context to reduce false signals.
# Targets 20-30 trades/year to minimize fee drag.

name = "4h_KAMA_Trend_Filter_v3"
timeframe = "4h"
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
    
    # Get 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on daily close
    # Efficiency ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[29] = close_1d[29]  # Seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    
    # 4x ATR for volatility regime filter (avoid chop)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First value
    atr = np.zeros_like(close)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]
    atr_mean = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_4h[i]) or np.isnan(atr_mean[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volatility filter: only trade when ATR is above average (trending market)
            vol_filter = atr[i] > atr_mean[i]
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: price above KAMA + volume + volatility filter
            if close[i] > kama_4h[i] and vol_filter and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + volume + volatility filter
            elif close[i] < kama_4h[i] and vol_filter and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or volatility drops
            if close[i] < kama_4h[i] or atr[i] < atr_mean[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or volatility drops
            if close[i] > kama_4h[i] or atr[i] < atr_mean[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals