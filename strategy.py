#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter (1w trend)
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
# Combined with RSI for momentum and 1w chop filter to avoid ranging markets.
# Works in bull markets via KAMA up + RSI > 50, in bear via KAMA down + RSI < 50.
# Chop filter ensures we only trade when market is trending (Chop < 38.2).
# Target: 7-25 trades/year (30-100 total over 4 years) for 1d timeframe.

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Chop on 1w: Chop = 100 * log10(sum(ATR(1)) / (HHV - LLV)) / log10(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.zeros(len(high_1w))
    tr1[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(high_1w)):
        tr1[i] = max(high_1w[i] - low_1w[i], 
                     abs(high_1w[i] - close_1w[i-1]), 
                     abs(low_1w[i] - close_1w[i-1]))
    
    # ATR(1) for 1w (simple average of TR)
    atr1 = tr1.copy()
    
    # Sum of ATR(1) over 14 periods
    sum_atr1 = np.zeros(len(high_1w))
    for i in range(13, len(high_1w)):
        sum_atr1[i] = np.sum(tr1[i-13:i+1])
    
    # Highest High and Lowest Low over 14 periods
    highest_high = np.zeros(len(high_1w))
    lowest_low = np.zeros(len(low_1w))
    for i in range(13, len(high_1w)):
        highest_high[i] = np.max(high_1w[i-13:i+1])
        lowest_low[i] = np.min(low_1w[i-13:i+1])
    
    # Chop calculation
    chop = np.full(len(high_1w), np.nan)
    for i in range(13, len(high_1w)):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(sum_atr1[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
    
    # Chop < 38.2 indicates trending market
    chop_trending = chop < 38.2
    chop_trending_aligned = align_htf_to_ltf(prices, df_1w, chop_trending)
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    
    # Pad arrays for alignment
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with first close
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # First average
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rsi = np.zeros_like(close)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan  # First 14 values are undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check chop filter (trending market)
        chop_ok = chop_trending_aligned[i]
        
        if position == 1:  # Long position
            # Exit: KAMA turns down or RSI < 40
            if kama[i] < kama[i-1] or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: KAMA turns up or RSI > 60
            if kama[i] > kama[i-1] or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if chop_ok:
                # Strong upward momentum + KAMA up
                if kama[i] > kama[i-1] and rsi[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Strong downward momentum + KAMA down
                elif kama[i] < kama[i-1] and rsi[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals