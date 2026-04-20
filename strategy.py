#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_1d_Chop_Filter
Hypothesis: KAMA trend direction on 4h + RSI momentum filter + 1d chop regime filter.
Long when KAMA trending up, RSI > 50, and 1d chop > 61.8 (ranging market).
Short when KAMA trending down, RSI < 50, and 1d chop > 61.8.
Uses chop to avoid trending markets where mean reversion fails.
Works in bull/bear: adapts to ranging conditions with mean reversion logic.
Target: 80-150 total trades over 4 years (20-38/year) with position size 0.25.
"""

name = "4h_KAMA_Trend_RSI_1d_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d chopiness index (EHLERS)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        atr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # True range sum over 14 periods
    tr_sum_14 = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i >= 13:
            tr_sum_14[i] = np.sum(atr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.zeros(len(close_1d))
    ll_14 = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i >= 13:
            hh_14[i] = np.max(high_1d[i-13:i+1])
            ll_14[i] = np.min(low_1d[i-13:i+1])
    
    # Chopiness index
    chop_1d = np.full(len(close_1d), 50.0)
    for i in range(len(close_1d)):
        if i >= 13 and tr_sum_14[i] > 0 and hh_14[i] > ll_14[i]:
            log_val = np.log10(tr_sum_14[i] / (hh_14[i] - ll_14[i]))
            chop_1d[i] = 100 * log_val / np.log10(14)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate KAMA on 4h
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=9))  # 9-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else np.zeros(len(close))
    volatility = np.concatenate([[np.nan], volatility[:-1]])  # shift for alignment
    
    er = np.zeros(len(close))
    for i in range(len(close)):
        if i >= 9 and not np.isnan(volatility[i]) and volatility[i] > 0:
            er[i] = direction[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.zeros(len(close))
    for i in range(len(close)):
        if i >= 9:
            sc[i] = (er[i] * (0.6667 - 0.0645) + 0.0645) ** 2
        else:
            sc[i] = 0
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI on 4h
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.zeros(len(close))
    for i in range(len(close)):
        if i >= 14 and avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        elif i >= 14 and avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rsi[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if chop data not available
        if np.isnan(chop_1d_aligned[i]):
            continue
            
        # Only trade in ranging markets (chop > 61.8)
        if chop_1d_aligned[i] <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # In ranging market: mean reversion at extremes
        if position == 0:
            # Long when KAMA trending up and RSI > 50 (bullish momentum)
            if kama[i] > kama[i-1] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short when KAMA trending down and RSI < 50 (bearish momentum)
            elif kama[i] < kama[i-1] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI < 50
            if kama[i] < kama[i-1] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI > 50
            if kama[i] > kama[i-1] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals