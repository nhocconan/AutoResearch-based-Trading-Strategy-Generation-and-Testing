#!/usr/bin/env python3
"""
1d_1w_KAMA_Regime_VolumeBreakout_V1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with weekly pivot bias and volume spike confirmation. Enter long when price > KAMA
in bullish weekly regime with volume expansion; short when price < KAMA in bearish weekly regime.
Weekly regime filter avoids counter-trend trades in strong weekly trends. Low frequency (~15-25 trades/year)
to minimize fee drag. Works in bull/bear: KAMA adapts to trend speed, weekly pivot provides structural bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for KAMA and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Kaufman Adaptive Moving Average (KAMA) with ER=10, fast=2, slow=30
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close_1d, 10))  # |close[t] - close[t-10]|
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
    
    er = np.zeros_like(close_1d)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)      # EMA(2)
    slow_sc = 2 / (30 + 1)     # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[29] = close_1d[29]  # seed at period 30
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.zeros_like(close_1d)
    tr2 = np.zeros_like(close_1d)
    tr3 = np.zeros_like(close_1d)
    tr1[1:] = high_1d[1:] - low_1d[1:]
    tr2[1:] = np.abs(high_1d[1:] - close_1d[:-1])
    tr3[1:] = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close_1d)
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Load 1w data for weekly pivot bias (long-term direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1 = 2 * pivot_point - prev_low_1w
    s1 = 2 * pivot_point - prev_high_1w
    
    # Weekly bias: price above pivot = bullish, below = bearish
    weekly_bias = align_htf_to_ltf(prices, df_1w, pivot_point)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(weekly_bias[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter: bullish if close > pivot, bearish if close < pivot
        weekly_close = align_htf_to_ltf(prices, df_1w, close_1w)
        weekly_bullish = not np.isnan(weekly_close[i]) and weekly_close[i] > weekly_bias[i]
        weekly_bearish = not np.isnan(weekly_close[i]) and weekly_close[i] < weekly_bias[i]
        
        if position == 0:
            # Long: price > KAMA (bullish momentum) AND weekly bullish bias AND volume spike
            if (price > kama_aligned[i] and 
                weekly_bullish and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (bearish momentum) AND weekly bearish bias AND volume spike
            elif (price < kama_aligned[i] and 
                  weekly_bearish and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA (trend reversal) OR weekly bias turns bearish
            if (price < kama_aligned[i] or 
                (not np.isnan(weekly_close[i]) and weekly_close[i] < weekly_bias[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA (trend reversal) OR weekly bias turns bullish
            if (price > kama_aligned[i] or 
                (not np.isnan(weekly_close[i]) and weekly_close[i] > weekly_bias[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Regime_VolumeBreakout_V1"
timeframe = "1d"
leverage = 1.0