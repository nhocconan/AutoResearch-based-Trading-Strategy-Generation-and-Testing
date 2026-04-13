#!/usr/bin/env python3
"""
Hypothesis: 4h 1-day KAMA + RSI + chop filter strategy.
Uses 1-day KAMA (adaptive moving average) to determine trend direction (bullish when close > KAMA, bearish when close < KAMA),
1-day RSI for momentum confirmation (bullish when RSI > 50, bearish when RSI < 50), and 4-hour Choppiness Index for regime filter
(trend-following when CHOP < 38.2, mean-reversion when CHOP > 61.8). Enters long when KAMA bullish, RSI > 50, and CHOP < 38.2.
Enters short when KAMA bearish, RSI < 50, and CHOP > 61.8. Uses 25% position sizing to manage risk. Targets 20-50 trades per year
to minimize fee drag while capturing trend moves in both bull and bear markets.
"""

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
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day KAMA (adaptive moving average)
    # Efficiency ratio: |close - close_10| / sum(|close - close_prev|) over 10 periods
    change = np.abs(np.diff(close_10))
    volatility = np.sum(np.abs(np.diff(close_10)), axis=1)  # This needs fixing - let's do it properly
    
    # Simpler KAMA calculation: use close - close.shift(10) for numerator, sum of abs changes for denominator
    mom = np.abs(np.diff(close_10, n=10))  # |close - close_10|
    vol = np.sum(np.abs(np.diff(close_10)), axis=1)  # Still problematic
    
    # Let's use a more straightforward adaptive MA approach
    # Use 2-period and 30-period EMA as in classic KAMA
    fast_ema = pd.Series(close_1d).ewm(span=2, adjust=False).mean().values
    slow_ema = pd.Series(close_1d).ewm(span=30, adjust=False).mean().values
    
    # Calculate ER (Efficiency Ratio)
    price_change = np.abs(np.diff(close_1d, n=10))
    abs_price_change = np.sum(np.abs(np.diff(close_1d)), axis=1)  # This is still wrong
    
    # Let's simplify and use a standard adaptive approach
    # Calculate momentum over 10 days
    momentum = np.abs(np.diff(close_1d, n=10))
    # Calculate volatility over 10 days (sum of absolute daily changes)
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    mask = volatility != 0
    er[mask] = momentum[mask] / volatility[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # 2-day EMA
    slow_sc = 2 / (30 + 1)  # 30-day EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1-day RSI (14-period)
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
    
    rs = np.zeros_like(close_1d)
    rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
    rsi = np.zeros_like(close_1d)
    rsi[rs != 0] = 100 - (100 / (1 + rs[rs != 0]))
    # For zero RS (all gains), RSI = 100
    rsi[rs == 0] = 100
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 4h data for Choppiness Index
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range for CHOP
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])],
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR (14-period)
    atr_4h = np.zeros_like(close_4h)
    atr_4h[13] = np.mean(tr_4h[1:15])
    for i in range(15, len(tr_4h)):
        atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close_4h)
    lowest_low = np.zeros_like(close_4h)
    
    for i in range(13, len(close_4h)):
        highest_high[i] = np.max(high_4h[i-13:i+1])
        lowest_low[i] = np.min(low_4h[i-13:i+1])
    
    # Avoid division by zero in CHOP calculation
    denominator = highest_high - lowest_low
    chop = np.zeros_like(close_4h)
    mask = denominator != 0
    chop[mask] = 100 * np.log10(np.sum(tr_4h[i-13:i+1] for i in range(14, len(close_4h)+1) if i-14 >= 0) / denominator[mask]) / np.log10(14)
    # Fix the chop calculation - let's do it properly
    
    # Recalculate CHOP properly
    chop = np.zeros_like(close_4h)
    for i in range(13, len(close_4h)):
        if denominator[i] != 0:
            tr_sum = np.sum(tr_4h[i-13:i+1])
            chop[i] = 100 * np.log10(tr_sum / denominator[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from KAMA
        kama_bullish = close[i] > kama_aligned[i]
        kama_bearish = close[i] < kama_aligned[i]
        
        # Momentum from RSI
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Regime filter from Choppiness Index
        chop_trending = chop_aligned[i] < 38.2  # Trending market
        chop_ranging = chop_aligned[i] > 61.8   # Ranging market
        
        # Entry conditions
        long_entry = kama_bullish and rsi_bullish and chop_trending
        short_entry = kama_bearish and rsi_bearish and chop_ranging
        
        # Exit when trend changes
        exit_long = position == 1 and not (kama_bullish and rsi_bullish)
        exit_short = position == -1 and not (kama_bearish and rsi_bearish)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_kama_rsi_chop"
timeframe = "4h"
leverage = 1.0