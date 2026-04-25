#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter (1d timeframe)
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing a reliable trend filter.
In trending markets (chop < 61.8), KAMA acts as dynamic support/resistance. RSI(14) filters extremes.
Long when price > KAMA and RSI < 70 (avoid overbought), short when price < KAMA and RSI > 30 (avoid oversold).
Chop regime filter ensures we only trade in trending conditions, avoiding whipsaws in ranging markets.
Works in both bull (buy dips to KAMA in uptrend) and bear (sell rallies to KAMA in downtrend) via symmetric logic.
Target 15-25 trades/year on 1d to minimize fee drag.
"""

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
    
    # Get 1w data for chop regime filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_1w = np.zeros(len(df_1w))
    tr_1w = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    for i in range(14, len(df_1w)):
        atr_1w[i] = np.mean(tr_1w[i-13:i+1])
    
    highest_high_1w = np.zeros(len(df_1w))
    lowest_low_1w = np.zeros(len(df_1w))
    for i in range(len(df_1w)):
        if i < 14:
            highest_high_1w[i] = np.nan
            lowest_low_1w[i] = np.nan
        else:
            highest_high_1w[i] = np.max(high_1w[i-13:i+1])
            lowest_low_1w[i] = np.min(low_1w[i-13:i+1])
    
    chop_1w = np.zeros(len(df_1w))
    for i in range(14, len(df_1w)):
        if atr_1w[i] > 0 and (highest_high_1w[i] - lowest_low_1w[i]) > 0:
            chop_1w[i] = 100 * np.log10(sum(atr_1w[i-13:i+1]) / (highest_high_1w[i] - lowest_low_1w[i])) / np.log10(14)
        else:
            chop_1w[i] = np.nan
    
    # Align chop to 1d timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate KAMA on 1d close
    close_1d = pd.Series(close)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Fix: volatility needs to be rolling sum of absolute daily changes
    volatility_roll = np.zeros_like(close)
    for i in range(10, n):
        volatility_roll[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.zeros(n)
    er[10:] = change[9:] / volatility_roll[10:]
    er[er == 0] = 0.000001  # avoid division by zero
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[:9] = close[:9]
    for i in range(9, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.zeros(n)
    rs[14:] = avg_gain[14:] / np.where(avg_loss[14:] == 0, 0.000001, avg_loss[14:])
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for KAMA, RSI, chop
    start_idx = max(30, 14, 14)  # KAMA needs ~30, RSI 14, chop 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(chop_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop_1w_aligned[i]
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Look for entry signals
            # Long: price > KAMA, RSI < 70 (not overbought), trending regime
            long_entry = (curr_close > kama_val) and (rsi_val < 70) and trending_regime
            # Short: price < KAMA, RSI > 30 (not oversold), trending regime
            short_entry = (curr_close < kama_val) and (rsi_val > 30) and trending_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price < KAMA OR RSI > 70 (overbought) OR chop > 61.8 (ranging)
            if curr_close < kama_val or rsi_val > 70 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price > KAMA OR RSI < 30 (oversold) OR chop > 61.8 (ranging)
            if curr_close > kama_val or rsi_val < 30 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime_TrendFilter"
timeframe = "1d"
leverage = 1.0