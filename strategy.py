#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction with RSI overbought/oversold and Choppiness Index regime filter. 
Enter long when KAMA turns up and RSI < 30 in trending market (CHOP < 38.2); short when KAMA turns down and RSI > 70 in trending market. 
Use 1d timeframe for Choppiness Index to avoid whipsaw in ranging markets. Designed for low trade frequency (<25/year) to minimize fee flood and work in both bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA calculation (ER=10, fast=2, slow=30) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.diff(volatility, prepend=volatility[0])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily data for Choppiness Index ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # first period
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max and min close over 14 periods
    max_close = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if sum_atr[i] > 0 and max_close[i] != min_close[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align daily Choppiness Index to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: covers KAMA initialization, RSI, ATR calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # KAMA direction: slope over 3 periods
        kama_up = kama[i] > kama[i-3]
        kama_down = kama[i] < kama[i-3]
        
        # Choppiness regime: trending if CHOP < 38.2
        trending = chop_aligned[i] < 38.2
        
        # Entry conditions
        if position == 0:
            # Long: KAMA turning up + RSI oversold + trending market
            if kama_up and rsi[i] < 30 and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA turning down + RSI overbought + trending market
            elif kama_down and rsi[i] > 70 and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal at opposite RSI extreme or KAMA reversal
        elif position == 1:
            if rsi[i] > 70 or not kama_up:  # RSI overbought or KAMA turns down
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if rsi[i] < 30 or not kama_down:  # RSI oversold or KAMA turns up
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0