#!/usr/bin/env python3
"""
1d_KAMA_1dRSI_TrendFilter_V2
Strategy: KAMA direction + RSI + chop filter on daily timeframe.
Long: KAMA direction up + RSI(14) > 50 + Choppiness Index < 61.8 (trending)
Short: KAMA direction down + RSI(14) < 50 + Choppiness Index < 61.8 (trending)
Exit: Opposite KAMA direction signal
Position size: 0.25
Uses daily KAMA for trend, RSI for momentum filter, Choppiness Index for regime filter.
Avoids range-bound whipsaws by requiring trending conditions (CHOP < 61.8).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = |close - close[10]| / sum(|close - close[-1]| for 10 periods)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev KAMA + SC * (close - prev KAMA)
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(change[i-9:i+1])
    
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if volatility[i] > 0:
            er[i] = np.abs(close_1d[i] - close_1d[i-10]) / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on daily data
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    
    atr14 = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr14[i] = np.mean(tr[1:15])
        else:
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    sum_atr14 = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        sum_atr14[i] = np.sum(atr14[i-13:i+1])
    
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        max_high[i] = np.max(high_1d[i-14:i+1])
        min_low[i] = np.min(low_1d[i-14:i+1])
    
    chop = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(sum_atr14[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(30, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters
        kama_up = close[i] > kama_aligned[i]
        kama_down = close[i] < kama_aligned[i]
        rsi_filter_long = rsi_aligned[i] > 50
        rsi_filter_short = rsi_aligned[i] < 50
        chop_filter = chop_aligned[i] < 61.8  # trending market
        
        # Entry conditions
        if position == 0:
            # Long: KAMA up + RSI > 50 + trending
            if kama_up and rsi_filter_long and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + trending
            elif kama_down and rsi_filter_short and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA down signal
            if kama_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up signal
            if kama_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_1dRSI_TrendFilter_V2"
timeframe = "1d"
leverage = 1.0