#!/usr/bin/env python3
name = "12h_KAMA_Direction_RSI_Chop"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA direction from daily close
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Handle array shapes
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) from daily close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[13] = np.nanmean(gain[1:14])
    avg_loss[13] = np.nanmean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index from daily high/low/close
    atr_period = 14
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([np.full(1, np.nan), tr])
    atr = np.full_like(close_1d, np.nan)
    atr[13] = np.nanmean(tr[1:15])
    for i in range(15, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of true ranges over period
    sum_tr = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        sum_tr[i] = np.sum(tr[i-13:i+1])
    
    # Max high - min low over period
    max_high = np.full_like(close_1d, np.nan)
    min_low = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if sum_tr[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    
    # Align Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Wait for KAMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, chop < 61.8 (trending)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, chop < 61.8 (trending)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA or RSI < 50 or chop > 61.8 (ranging)
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 50 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA or RSI > 50 or chop > 61.8 (ranging)
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 50 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA direction with RSI filter and chop regime filter on 12h timeframe
# - KAMA adapts to market conditions, reducing whipsaws in ranging markets
# - RSI > 50 for long, < 50 for short ensures momentum alignment
# - Chop < 61.8 filters for trending markets only (avoids ranging whipsaws)
# - Works in bull markets (KAMA up + RSI > 50) and bear markets (KAMA down + RSI < 50)
# - Chop filter prevents trades during sideways consolidation
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Daily timeframe for indicators provides stability and reduces noise