#!/usr/bin/env python3
"""
Daily KAMA + RSI + Chop Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies adaptive trend,
RSI confirms momentum, and Choppiness Index filters ranging markets.
Works in both bull and bear by only taking trend-following signals in strong trends.
Designed for daily timeframe to keep trade frequency low (target: 7-25 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(er_period, len(close)):
        if volatility[i-er_period:i+1].sum() > 0:
            er[i] = change[i-er_period:i+1].sum() / volatility[i-er_period:i+1].sum()
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr = np.zeros(len(close))
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    # True Range sum
    tr_sum = np.zeros(len(close))
    for i in range(period, len(close)):
        tr_sum[i] = tr[i-period+1:i+1].sum()
    
    # Highest high and lowest low over period
    max_high = np.zeros(len(close))
    min_low = np.zeros(len(close))
    for i in range(period-1, len(close)):
        max_high[i] = high[i-period+1:i+1].max()
        min_low[i] = low[i-period+1:i+1].min()
    
    # Chop calculation
    chop = np.full(len(close), 50.0)
    for i in range(period-1, len(close)):
        if max_high[i] != min_low[i] and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily KAMA for trend
    kama = calculate_kama(close, er_period=10, fast=2, slow=30)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = gain[1:i+1].mean()
            avg_loss[i] = loss[1:i+1].mean()
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly Chop for regime filter
    df_1w = get_htf_data(prices, '1w')
    chop = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when trending (Chop < 38.2) or extreme ranging (Chop > 61.8 for mean reversion)
        # But we'll use Chop < 38.2 for trend following only
        is_trending = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: trend weakness or RSI overbought
            if close[i] <= kama[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakness or RSI oversold
            if close[i] >= kama[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in trending markets (Chop < 38.2)
            if is_trending:
                # Enter long: price above KAMA and RSI rising from oversold
                if close[i] > kama[i] and rsi[i] > 30 and rsi[i] < 50 and rsi[i] > rsi[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price below KAMA and RSI falling from overbought
                elif close[i] < kama[i] and rsi[i] < 70 and rsi[i] > 50 and rsi[i] < rsi[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals