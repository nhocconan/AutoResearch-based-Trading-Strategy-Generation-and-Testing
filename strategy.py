#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter_v3
# Hypothesis: 1-day KAMA trend + RSI(14) overbought/oversold + Choppiness Index(14) regime filter.
# Long when KAMA rising, RSI<40, CHOP>61.8 (range). Short when KAMA falling, RSI>60, CHOP>61.8.
# Works in ranging markets by mean-reverting at RSI extremes with trend filter.
# In strong trends (CHOP<38.2), avoids trades to prevent whipsaw.
# Target: 10-20 trades/year per symbol.

name = "1d_KAMA_RSI_Chop_Filter_v3"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (1-day)
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |diff| over window
    # Handle first 9 values where diff(n=10) is invalid
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    ER = np.where(volatility != 0, change / volatility, 0)
    # Smooth constants
    sc = (ER * 0.2 + (1 - ER) * 0.064) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # First 14 values are NaN due to min_periods
    
    # Calculate Choppiness Index(14) from weekly data
    # CHOP = 100 * log10(sum(ATR(1)) / (highest_high - lowest_low)) / log10(14)
    tr1 = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1_sum / (highest_high - lowest_low)) / np.log10(14)
    # Align weekly CHOP to daily
    chop_1w = pd.Series(chop).values
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 10)  # RSI and KAMA seed
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (trend up), RSI oversold, choppy market
            if (kama[i] > kama[i-1] and 
                rsi[i] < 40 and 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (trend down), RSI overbought, choppy market
            elif (kama[i] < kama[i-1] and 
                  rsi[i] > 60 and 
                  chop_1w_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA falling OR RSI overbought OR market becomes trending
            if (kama[i] < kama[i-1] or 
                rsi[i] > 60 or 
                chop_1w_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA rising OR RSI oversold OR market becomes trending
            if (kama[i] > kama[i-1] or 
                rsi[i] < 40 or 
                chop_1w_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals