#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend with RSI and Chop filter on 1d timeframe.
# Uses 1d KAMA (efficiency ratio smoothing) for trend, 1d RSI(14) for momentum,
# and 1d Choppiness Index for regime detection.
# Long: KAMA rising, RSI > 50, Chop < 61.8 (trending regime)
# Short: KAMA falling, RSI < 50, Chop < 61.8 (trending regime)
# Avoids choppy markets where trend strategies fail.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # EMA(2) for fast
    slow_sc = 2 / (30 + 1) # EMA(30) for slow
    
    # Calculate Efficiency Ratio and KAMA
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros(n)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.zeros(n)
    tr[1:] = np.maximum(tr1, np.maximum(tr2, tr3))
    
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    chop = np.zeros(n)
    for i in range(14, n):
        if atr[i] != 0 and max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i-13:i+1]) / (atr[i] * (max_high[i] - min_low[i]))) / np.log10(14)
        else:
            chop[i] = 50
    
    # Signals
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position
    
    for i in range(14, n):
        # Skip if any required data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            i == 0):  # need previous KAMA for trend
            signals[i] = 0.0
            continue
        
        kama_trend = kama[i] > kama[i-1]  # Rising KAMA = uptrend
        rsi_bull = rsi[i] > 50
        rsi_bear = rsi[i] < 50
        trending_regime = chop[i] < 61.8  # Chop < 61.8 = trending
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, trending regime
            if kama_trend and rsi_bull and trending_regime:
                position = 1
                signals[i] = position_size
            # Short: KAMA falling, RSI < 50, trending regime
            elif not kama_trend and rsi_bear and trending_regime:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA falling or Chop > 61.8 (choppy)
            if not kama_trend or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA rising or Chop > 61.8 (choppy)
            if kama_trend or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_Chop_Trend"
timeframe = "1d"
leverage = 1.0