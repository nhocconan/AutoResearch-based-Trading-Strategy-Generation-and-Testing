#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI for momentum strength, and Choppiness Index for regime filtering. Only take long when
KAMA is rising, RSI > 50, and market is trending (CHOP < 38.2); short when KAMA falling,
RSI < 50, and trending. This avoids whipsaws in ranging markets and captures sustained
trends in both bull and bear markets. Low trade frequency expected (<25/year) to minimize
fee drag, with weekly timeframe used for regime confirmation.
"""

name = "1d_KAMA_Trend_RSI_ChopFilter"
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
    
    # Get weekly data for regime filter (choppiness index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === DAILY INDICATORS ===
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    # Parameters: ER length=10, Fast SC=2, Slow SC=30
    er_len = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first er_len elements
    er = np.full_like(change, np.nan, dtype=np.float64)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing Constants
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.where(np.isnan(sc), 0, sc)  # where er is nan
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # seed
    for i in range(er_len + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (Relative Strength Index) - momentum
    rsi_len = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    if len(gain) >= rsi_len:
        avg_gain[rsi_len] = np.mean(gain[:rsi_len])
        avg_loss[rsi_len] = np.mean(loss[:rsi_len])
        for i in range(rsi_len + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_len - 1) + gain[i-1]) / rsi_len
            avg_loss[i] = (avg_loss[i-1] * (rsi_len - 1) + loss[i-1]) / rsi_len
    
    rsi = np.full_like(close, 50.0)  # default to neutral
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 100)
    
    # === WEEKLY INDICATORS (for regime filter) ===
    # Choppiness Index - determines if market is trending or ranging
    chop_len = 14
    atr_1w = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        tr = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
        if i == 1:
            atr_1w[i] = tr
        else:
            atr_1w[i] = (atr_1w[i-1] * (chop_len - 1) + tr) / chop_len
    
    # Sum of True Ranges over chop_len periods
    sum_tr = np.full_like(close_1w, np.nan)
    if len(atr_1w) >= chop_len:
        sum_tr[chop_len-1] = np.sum(atr_1w[:chop_len])
        for i in range(chop_len, len(close_1w)):
            sum_tr[i] = sum_tr[i-1] - atr_1w[i-chop_len] + atr_1w[i]
    
    # Highest high and lowest low over chop_len periods
    max_hh = np.full_like(close_1w, np.nan)
    min_ll = np.full_like(close_1w, np.nan)
    if len(high_1w) >= chop_len:
        max_hh[chop_len-1] = np.max(high_1w[:chop_len])
        min_ll[chop_len-1] = np.min(low_1w[:chop_len])
        for i in range(chop_len, len(high_1w)):
            max_hh[i] = max(max_hh[i-1], high_1w[i])
            min_ll[i] = min(min_ll[i-1], low_1w[i])
    
    # Choppiness Index
    chop = np.full_like(close_1w, 50.0)  # default to neutral
    denominator = max_hh - min_ll
    mask = (denominator > 0) & (~np.isnan(sum_tr))
    chop[mask] = 100 * np.log10(sum_tr[mask] / denominator[mask]) / np.log10(chop_len)
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)  # Wait for weekly close
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)  # Chop uses current week
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_len + 1, rsi_len + 1, chop_len)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising (price > KAMA), RSI > 50, trending market (CHOP < 38.2)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling (price < KAMA), RSI < 50, trending market (CHOP < 38.2)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 40 OR market becomes ranging (CHOP > 61.8)
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] < 40 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 60 OR market becomes ranging (CHOP > 61.8)
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] > 60 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals