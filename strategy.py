#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Chop
Hypothesis: KAMA trend direction on 12h combined with RSI and Choppiness index for regime filtering. 
Works in bull markets by capturing trends and in bear markets by avoiding false signals via chop filter.
Designed for 15-25 trades/year per symbol with strict entry conditions to minimize fee drag.
"""
name = "12h_KAMA_Direction_RSI_Chop"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Choppiness index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop = np.concatenate([[np.nan] * 13, chop[13:]])  # align with 1d index
    
    # Align Chop to 12h timeframe with 1-bar delay (ensure previous day's chop is used)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 12h data for KAMA and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(np.concatenate([[np.nan], np.diff(close_12h)]))
    sum_abs_change = pd.Series(change).rolling(window=er_len, min_periods=er_len).sum().values
    er = np.where(sum_abs_change > 0, np.abs(np.concatenate([[np.nan] * (er_len-1), np.diff(close_12h[er_len-1:])])) / sum_abs_change, 0)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[er_len] = close_12h[er_len]  # seed
    for i in range(er_len + 1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (no additional delay needed as it's based on current bar)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # RSI(14) on 12h
    delta = np.concatenate([[np.nan], np.diff(close_12h)])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi[14:]])  # align
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend), RSI > 50, Chop < 61.8 (trending market)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 61.8 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend), RSI < 50, Chop < 61.8 (trending market)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: opposite condition
            if position == 1:
                if close[i] <= kama_aligned[i] or rsi_aligned[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= kama_aligned[i] or rsi_aligned[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals