#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_and_Chop_Filter
Hypothesis: On daily timeframe, KAMA adapts to market efficiency, providing a robust trend filter. 
Combined with RSI for momentum confirmation and Choppiness Index to avoid ranging markets, 
this strategy captures strong trending moves while minimizing false signals in chop. 
Works in both bull and bear by following KAMA direction only when market is trending (CHOP < 38.2) 
and momentum confirms (RSI > 50 for long, < 50 for short).
"""

name = "1d_KAMA_Trend_With_RSI_and_Chop_Filter"
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
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on daily close (using close prices directly)
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no alignment needed as already daily)
    kama_aligned = kama  # Already on daily timeframe
    
    # Calculate RSI(14) on daily
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index on weekly data
    # CHOP = 100 * log10(sum(TR, n) / (max(HH, n) - min(LL, n))) / log10(n)
    # Using weekly data for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if np.isnan(tr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]) or hh[i] == ll[i]:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
    
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA (uptrend), RSI > 50, and CHOP < 38.2 (trending market)
            if (close[i] > kama_aligned[i] and 
                rsi[i] > 50 and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend), RSI < 50, and CHOP < 38.2 (trending market)
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] < 50 and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA (trend change)
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA (trend change)
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals