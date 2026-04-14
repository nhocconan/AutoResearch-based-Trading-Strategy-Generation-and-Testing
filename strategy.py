#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day KAMA + RSI + chop regime filter.
# Long when KAMA turns up (bullish), RSI < 50 (avoid overbought), and chop > 61.8 (range) for mean reversion.
# Short when KAMA turns down (bearish), RSI > 50 (avoid oversold), and chop > 61.8 (range) for mean reversion.
# Exit when KAMA reverses or chop < 38.2 (trending) to avoid whipsaw in trends.
# KAMA adapts to market noise, RSI avoids extremes, chop filter ensures mean-reversion logic only applies in ranging markets.
# Designed to work in both bull and bear markets by fading extremes in ranging conditions.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for KAMA, RSI, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for KAMA/RSI/chop
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10, 2, 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will fix below
    # Proper ER calculation
    price_diff = np.abs(np.diff(close_1d, k=10))  # 10-period net change
    abs_diff = np.abs(np.diff(close_1d, k=1))    # 1-period changes
    # Sum of absolute changes over 10 periods
    volatility_sum = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility_sum[i] = np.sum(abs_diff[i-9:i+1])
    er = np.zeros_like(close_1d)
    er[10:] = price_diff[10:] / volatility_sum[10:]
    er[volatility_sum[10:] == 0] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # start at index 9
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi])
    
    # Calculate Chop (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # ATR
    atr = np.zeros_like(close_1d)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # Max/Min over 14 periods
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        max_high[i] = np.max(high_1d[i-12:i+1])
        min_low[i] = np.min(low_1d[i-12:i+1])
    # Chop calculation
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(atr[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # avoid division by zero
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # Need KAMA/RSI/chop periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI filters: avoid extremes
        rsi_not_overbought = rsi_aligned[i] < 50
        rsi_not_oversold = rsi_aligned[i] > 50
        
        # Chop regime: only trade in ranging markets (chop > 61.8)
        ranging = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2  # exit signal
        
        if position == 0:
            # Look for mean reversion entries in ranging market
            # Long: KAMA turning up AND RSI not overbought AND ranging
            if (kama_up and 
                rsi_not_overbought and 
                ranging):
                position = 1
                signals[i] = position_size
            # Short: KAMA turning down AND RSI not oversold AND ranging
            elif (kama_down and 
                  rsi_not_oversold and 
                  ranging):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA reverses down OR trend emerges (chop < 38.2)
            if (not kama_up or 
                trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA reverses up OR trend emerges (chop < 38.2)
            if (not kama_down or 
                trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_KAMA_RSI_Chop_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0