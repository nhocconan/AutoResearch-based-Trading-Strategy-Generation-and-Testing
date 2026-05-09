#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA adapts to market noise, reducing whipsaws in sideways markets.
# Combined with RSI filter to avoid overbought/oversold extremes, this strategy works
# in both bull and bear markets by only trading in the direction of the adaptive trend
# when momentum is not extreme. Uses 4h timeframe with 1d trend filter for higher reliability.

name = "4h_KAMA_Trend_With_RSI_Filter"
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
    
    # Calculate 1d KAMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) for KAMA
    er = np.full_like(close_1d, np.nan)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    for i in range(10, len(close_1d)):
        if i >= 10:
            abs_change = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            net_change = np.abs(close_1d[i] - close_1d[i-10])
            if abs_change > 0:
                er[i] = net_change / abs_change
    
    # Smoothing constants
    sc = np.full_like(close_1d, np.nan)
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    for i in range(len(close_1d)):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) > 0:
        kama_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            if not np.isnan(sc[i]):
                kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
            else:
                kama_1d[i] = kama_1d[i-1]
    
    # Align KAMA to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Ensure KAMA and RSI are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA AND RSI not overbought
            if (close[i] > kama_1d_aligned[i] and 
                rsi[i] < 70):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA AND RSI not oversold
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi[i] > 30):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI overbought
            if (close[i] < kama_1d_aligned[i] or 
                rsi[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI oversold
            if (close[i] > kama_1d_aligned[i] or 
                rsi[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals