#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_filter_v2
Hypothesis: Daily KAMA trend with RSI momentum and Choppiness index regime filter.
Enters long when KAMA upward, RSI > 50, and CHOP > 61.8 (range market) for mean reversion to upside.
Enters short when KAMA downward, RSI < 50, and CHOP > 61.8 for mean reversion to downside.
Uses weekly trend filter to avoid counter-trend trades in strong trends.
Designed to work in both bull (mean reversion in range) and bear (mean reversion in range) markets.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drift.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend direction
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, n):
        if i >= 10:
            change_val = np.abs(close[i] - close[i-10])
            volatility_val = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_val > 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initial average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder smoothing
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rsi = np.zeros_like(close)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Calculate Choppiness Index
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr14 = np.zeros_like(close)
    for i in range(14, n):
        atr14[i] = np.mean(tr[i-13:i+1])
    
    # Sum of ATR over 14 periods
    sum_atr14 = np.zeros_like(close)
    for i in range(14, n):
        sum_atr14[i] = np.sum(atr14[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close)
    for i in range(14, n):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(sum_atr14[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI momentum
        rsi_over_50 = rsi[i] > 50
        rsi_under_50 = rsi[i] < 50
        
        # Choppiness regime (range market)
        chop_high = chop[i] > 61.8  # Range/trading market
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions: KAMA direction + RSI + Chop + weekly trend alignment
        long_entry = kama_up and rsi_over_50 and chop_high and weekly_uptrend
        short_entry = kama_down and rsi_under_50 and chop_high and weekly_downtrend
        
        # Exit conditions: opposite signal or chop low (trending market)
        long_exit = not kama_up or not rsi_over_50 or chop[i] < 38.2
        short_exit = not kama_down or not rsi_under_50 or chop[i] < 38.2
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop_filter_v2"
timeframe = "1d"
leverage = 1.0