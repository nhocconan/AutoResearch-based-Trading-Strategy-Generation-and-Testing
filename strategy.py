#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_Spike_and_CHOP_Filter
# Hypothesis: KAMA identifies adaptive trend direction; volume spike confirms momentum; CHOP filter avoids ranging markets.
# Designed for 4h timeframe to work in both bull and bear markets by combining trend, momentum, and regime filters.
# Targets 20-40 trades/year to minimize fee drag while maintaining edge.

name = "4h_KAMA_Trend_With_Volume_Spike_and_CHOP_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for CHOP filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive trend) on 4h close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    # Recalculate volatility properly as rolling sum of absolute changes
    volatility = np.convolve(np.abs(np.diff(close, prepend=close[0])), np.ones(10), 'same')[:n]
    volatility[:9] = np.nan  # insufficient data for first 9
    ER = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    SC = (ER * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after 9 periods for volatility calculation
    for i in range(10, n):
        if not np.isnan(SC[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + SC[i] * (close[i] - kama[i-1])
    
    # Calculate CHOP on daily data
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # CHOP = 100 * log10(sum_tr / (ATR * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr / (atr * 14)) / np.log10(14)
    chop = chop[~np.isnan(chop)]  # align length
    # Align CHOP to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > volume_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), CHOP (need daily ATR 14), volume MA (20)
    start_idx = max(10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # CHOP filter: only trade when not ranging (CHOP < 50)
        not_ranging = chop_aligned[i] < 50
        
        # Volume spike confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long entry: price above KAMA + volume spike + not ranging
            if uptrend and vol_spike and not_ranging:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + volume spike + not ranging
            elif downtrend and vol_spike and not_ranging:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or chop rises (ranging) or volume dies
            if not uptrend or chop_aligned[i] >= 50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or chop rises or volume dies
            if not downtrend or chop_aligned[i] >= 50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals