#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_v2
# Hypothesis: Use KAMA trend direction on daily, RSI overbought/oversold for entry,
# and Choppiness Index to filter ranging vs trending markets. Only trade when
# Choppiness > 61.8 (ranging) and RSI is extreme, in direction of KAMA trend.
# Designed for low trade frequency (10-20/year) to avoid fee drag, works in
# ranging markets where mean reversion is effective.

name = "1d_KAMA_RSI_Chop_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d KAMA for trend direction
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=1))
    change = np.concatenate([[0], change])
    volatility = np.abs(np.diff(close, n=1))
    volatility = np.concatenate([[0], volatility])
    er = np.zeros_like(change)
    for i in range(1, len(change)):
        if np.sum(volatility[i-9:i+1]) > 0:
            er[i] = np.sum(change[i-9:i+1]) / np.sum(volatility[i-9:i+1])
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_trend = close > kama
    
    # 1w data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.absolute(high_1w[1:] - close_1w[:-1]), np.absolute(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[0], tr1])
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high - min_low)) / log10(14)
    chop = np.zeros_like(high_1w)
    for i in range(14, len(high_1w)):
        if max_high[i] - min_low[i] > 0:
            chop[i] = 100 * np.log10(np.sum(tr1[i-13:i+1]) / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align chop to 1d
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # 1d RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_trend[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop > 61.8 indicates ranging market (good for mean reversion)
        ranging = chop_aligned[i] > 61.8
        
        if position == 0 and ranging:
            # Long when RSI < 30 (oversold) and KAMA uptrend
            if rsi[i] < 30 and kama_trend[i]:
                signals[i] = 0.25
                position = 1
            # Short when RSI > 70 (overbought) and KAMA downtrend
            elif rsi[i] > 70 and not kama_trend[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (50) or trend changes
            if rsi[i] >= 50 or not kama_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (50) or trend changes
            if rsi[i] <= 50 or kama_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals