#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: Uses KAMA trend direction with RSI momentum and Choppiness Index regime filter on 1d timeframe.
# KAMA adapts to market noise, RSI identifies overbought/oversold, Choppiness filters trending vs ranging markets.
# Works in bull markets via trend following, in bear markets via mean reversion in ranging conditions.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1. KAMA (Kaufman Adaptive Moving Average) - 14-period
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros(n)
    er[9:] = change[9:] / np.maximum(volatility[9:], 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 2. RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros(n)
    rs[13:] = avg_gain[13:] / np.maximum(avg_loss[13:], 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 3. Choppiness Index (14-period) - gets weekly data for better regime detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate True Range for weekly data
    tr_1w = np.zeros(len(df_1w))
    tr_1w[0] = df_1w['high'].iloc[0] - df_1w['low'].iloc[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
    
    # Sum of TR over 14 periods
    tr_sum_14 = np.zeros(len(df_1w))
    for i in range(13, len(df_1w)):
        tr_sum_14[i] = np.sum(tr_1w[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high_14 = np.zeros(len(df_1w))
    min_low_14 = np.zeros(len(df_1w))
    for i in range(13, len(df_1w)):
        max_high_14[i] = np.max(df_1w['high'].iloc[i-13:i+1])
        min_low_14[i] = np.min(df_1w['low'].iloc[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros(len(df_1w))
    for i in range(13, len(df_1w)):
        if tr_sum_14[i] > 0 and (max_high_14[i] - min_low_14[i]) > 0:
            chop[i] = 100 * np.log10(tr_sum_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when undefined
    
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below KAMA OR RSI overbought (>70) OR market too choppy (>61.8)
            if (close[i] < kama[i] or 
                rsi[i] > 70 or 
                chop_aligned[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above KAMA OR RSI oversold (<30) OR market too choppy (>61.8)
            if (close[i] > kama[i] or 
                rsi[i] < 30 or 
                chop_aligned[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above KAMA AND RSI not overbought (<50) AND trending market (<38.2)
            if (close[i] > kama[i] and 
                rsi[i] < 50 and 
                chop_aligned[i] < 38.2):
                position = 1
                signals[i] = 0.25
            # Enter short: price below KAMA AND RSI not oversold (>50) AND trending market (<38.2)
            elif (close[i] < kama[i] and 
                  rsi[i] > 50 and 
                  chop_aligned[i] < 38.2):
                position = -1
                signals[i] = -0.25
    
    return signals