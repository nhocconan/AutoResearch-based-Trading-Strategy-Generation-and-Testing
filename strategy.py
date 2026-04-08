#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Hypothesis: Daily KAMA trend direction with RSI momentum and weekly chop regime filter.
# Long: KAMA rising (bullish trend) AND RSI > 50 (momentum) AND weekly chop < 61.8 (trending market)
# Short: KAMA falling (bearish trend) AND RSI < 50 (momentum) AND weekly chop < 61.8 (trending market)
# Exit: Opposite KAMA direction OR chop > 61.8 (ranging market) OR RSI extreme (>70 long exit, <30 short exit)
# Uses 1d primary timeframe with 1w HTF for chop regime to avoid look-ahead and reduce overtrading.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag in all market conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 10/2/30
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # 10-period sum of absolute changes
    # Pad arrays for alignment
    change_padded = np.full(n, np.nan)
    volatility_padded = np.full(n, np.nan)
    change_padded[10:] = change
    volatility_padded[10:] = volatility
    
    er = np.where(volatility_padded > 0, change_padded / volatility_padded, 0)
    # Limit ER to [0,1]
    er = np.clip(er, 0, 1)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # Seed
    avg_loss[13] = np.mean(loss[1:14])  # Seed
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1w data (14-period)
    chop_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        # True Range calculation
        tr1 = df_1w['high'].iloc[i] - df_1w['low'].iloc[i]
        tr2 = np.abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1])
        tr3 = np.abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over 14 periods
        tr_sum = 0
        for j in range(i-13, i+1):
            tr1_j = df_1w['high'].iloc[j] - df_1w['low'].iloc[j]
            tr2_j = np.abs(df_1w['high'].iloc[j] - df_1w['close'].iloc[j-1])
            tr3_j = np.abs(df_1w['low'].iloc[j] - df_1w['close'].iloc[j-1])
            tr_j = np.maximum(tr1_j, np.maximum(tr2_j, tr3_j))
            tr_sum += tr_j
        
        # Max high and min low over 14 periods
        max_high = np.max(df_1w['high'].iloc[i-13:i+1].values)
        min_low = np.min(df_1w['low'].iloc[i-13:i+1].values)
        
        if max_high != min_low:
            chop_1w[i] = 100 * np.log10(tr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1w[i] = 50  # neutral when no range
    
    # Align 1w chop to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        kama_val = kama[i]
        kama_prev = kama[i-1] if i > 0 else kama_val
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        price = close[i]
        
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Determine KAMA direction (rising/falling)
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 1:  # Long position
            # Exit conditions: KAMA falling OR chop > 61.8 (ranging) OR RSI > 70 (overbought)
            if kama_falling or chop_val > 61.8 or rsi_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: KAMA rising OR chop > 61.8 (ranging) OR RSI < 30 (oversold)
            if kama_rising or chop_val > 61.8 or rsi_val < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: KAMA direction + RSI momentum + trending market (chop < 61.8)
            if kama_rising and rsi_val > 50 and chop_val < 61.8:
                position = 1
                signals[i] = 0.25
            elif kama_falling and rsi_val < 50 and chop_val < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals