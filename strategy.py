#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Chop_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index (14) for regime filter.
Enter long when KAMA turns up, RSI > 50, and CHOP > 61.8 (ranging market); short when KAMA turns down,
RSI < 50, and CHOP > 61.8. Exit on opposite KAMA crossover. Weekly trend filter ensures alignment
with higher timeframe momentum. Designed for low trade frequency (<25/year) to minimize fee drag.
Works in ranging markets via mean reversion logic and avoids trending whipsaws via CHOP filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA(10,2,30) - fast, slow, lookback
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, kama_period))
    abs_price_change = np.sum(np.abs(np.diff(close_1d)), axis=0) if False else None  # placeholder
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(kama_period, len(close_1d)):
        if i >= kama_period:
            direction = np.abs(close_1d[i] - close_1d[i-kama_period])
            volatility = np.sum(np.abs(np.diff(close_1d[i-kama_period+1:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    
    # Smoothing constants
    sc = (er * (2/(slow_ema+1) - 2/(fast_ema+1)) + 2/(fast_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr_period = 14
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    atr = np.zeros_like(close_1d)
    for i in range(atr_period, len(close_1d)):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # True range for CHOP denominator
    tr_sum = np.zeros_like(close_1d)
    for i in range(atr_period, len(close_1d)):
        tr_sum[i] = np.sum(tr[i-atr_period+1:i+1])
    
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(atr_period, len(close_1d)):
        max_high[i] = np.max(high_1d[i-atr_period+1:i+1])
        min_low[i] = np.min(low_1d[i-atr_period+1:i+1])
    
    chop = np.zeros_like(close_1d)
    for i in range(atr_period, len(close_1d)):
        if tr_sum[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(atr_period)
        else:
            chop[i] = 50  # neutral
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend filter
    if len(close_1w) >= 21:
        ema_21_1w = np.zeros_like(close_1w)
        ema_21_1w[0] = close_1w[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_1w)):
            ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    else:
        ema_21_1w = np.full_like(close_1w, np.nan)
    
    # Align all indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction: slope of KAMA
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI conditions
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        
        # Choppiness filter: ranging market (CHOP > 61.8)
        chop_high = chop_aligned[i] > 61.8
        
        # Weekly trend filter: price above/below weekly EMA
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions
        long_setup = kama_up and rsi_above_50 and chop_high and price_above_weekly_ema
        short_setup = kama_down and rsi_below_50 and chop_high and price_below_weekly_ema
        
        # Exit on opposite KAMA crossover
        exit_long = kama_down
        exit_short = kama_up
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals