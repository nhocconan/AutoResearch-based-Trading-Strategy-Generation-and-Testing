#!/usr/bin/env python3
# 4h_KAMA_Trend_Filter
# Hypothesis: KAMA adapts to market noise, providing a reliable trend filter.
# In trending markets, KAMA follows price closely; in ranging markets, it flattens.
# Long when price > KAMA(10,2,30) and short when price < KAMA(10,2,30).
# Use 1d ADX > 20 to confirm trending regime and avoid false signals in low ADX.
# Add volume confirmation: current volume > 1.5 * 20-period average volume.
# Designed for 4h to balance trade frequency and reduce whipsaws.

name = "4h_KAMA_Trend_Filter"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        dir = np.abs(np.subtract(close, np.roll(close, er_length)))
        dir = np.where(np.arange(len(close)) < er_length, 0, dir)
        volatility = np.cumsum(change) - np.roll(np.cumsum(change), er_length)
        volatility = np.where(np.arange(len(close)) < er_length, 1, volatility)  # avoid div by zero
        er = np.where(volatility != 0, dir / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    # Minus Directional Movement
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def smoothed_avg(arr, period):
        avg = np.full_like(arr, np.nan)
        if len(arr) >= period:
            avg[period-1] = np.nanmean(arr[0:period])
            for i in range(period, len(arr)):
                avg[i] = (avg[i-1] * (period-1) + arr[i]) / period
        return avg
    
    atr = smoothed_avg(tr, 14)
    plus_di = 100 * smoothed_avg(plus_dm, 14) / atr
    minus_di = 100 * smoothed_avg(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smoothed_avg(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate KAMA and volume filter
    kama_val = kama(close, 10, 2, 30)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.nanmean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # ADX and volume MA ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_val[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        vol_conf = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price above KAMA, ADX > 20, volume confirmation
            if close[i] > kama_val[i] and adx_aligned[i] > 20 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, ADX > 20, volume confirmation
            elif close[i] < kama_val[i] and adx_aligned[i] > 20 and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR ADX < 20 (loss of trend)
            if close[i] < kama_val[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR ADX < 20 (loss of trend)
            if close[i] > kama_val[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals