#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_filter_v1
Hypothesis: Daily strategy using KAMA for trend direction, RSI for momentum confirmation,
and Choppiness Index as regime filter. Enters long when KAMA upward, RSI > 50, and choppy market (CHOP > 61.8).
Enters short when KAMA downward, RSI < 50, and choppy market. Uses choppy regime to avoid whipsaws in trends.
Designed to work in both bull and bear markets by focusing on mean-reversion in choppy conditions.
Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
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
    
    # Get weekly data for Choppiness Index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly Choppiness Index
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(np.roll(high_1w, 1) - close_1w)
    tr3 = np.abs(np.roll(low_1w, 1) - close_1w)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Align Choppiness Index to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate KAMA on daily close
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    abs_change = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # This needs fixing - let's do properly
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        direction = np.abs(close[i] - close[i-10])
        volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion
        choppy = chop_aligned[i] > 61.8
        
        # KAMA direction: slope of KAMA
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI momentum
        rsi_over_50 = rsi[i] > 50
        rsi_under_50 = rsi[i] < 50
        
        # Entry conditions
        long_entry = kama_rising and rsi_over_50 and choppy
        short_entry = kama_falling and rsi_under_50 and choppy
        
        # Exit conditions: opposite signal or choppy regime ends
        long_exit = not kama_rising or not choppy
        short_exit = not kama_falling or not choppy
        
        # Position sizing: smaller in choppy markets
        if choppy:
            position_size = 0.25
        else:
            position_size = 0.10  # Reduce size in trending markets
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0