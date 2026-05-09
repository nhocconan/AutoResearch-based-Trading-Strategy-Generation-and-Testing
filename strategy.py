#!/usr/bin/env python3
# 12h_KAMA_Trend_1dTrendFilter
# Strategy: Use KAMA on 12h to detect trend direction, confirmed by 1d EMA(50) filter
# Long when KAMA > price and price > 1d EMA50
# Short when KAMA < price and price < 1d EMA50
# Exit when trend reverses
# Designed for 12h timeframe with strong trend filtering to minimize trade frequency
# Uses volume confirmation to avoid false breakouts

name = "12h_KAMA_Trend_1dTrendFilter"
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
    volume = prices['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate KAMA(10,2,30) on 12h
    # ER (Efficiency Ratio) = abs(close - close[10]) / sum(abs(diff)) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))  # length n-10
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will compute properly below
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        price_change = abs(close[i] - close[i-10])
        volatility_sum = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if volatility_sum > 0:
            er[i] = price_change / volatility_sum
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 20-period average
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # KAMA needs warmup, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > average volume
        volume_ok = volume[i] > vol_ma[i]
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) and above 1d EMA50, with volume
            if close[i] > kama[i] and close[i] > ema_50_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) and below 1d EMA50, with volume
            elif close[i] < kama[i] and close[i] < ema_50_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend turns down (price < KAMA) or breaks 1d EMA50
            if close[i] < kama[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend turns up (price > KAMA) or breaks 1d EMA50
            if close[i] > kama[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals