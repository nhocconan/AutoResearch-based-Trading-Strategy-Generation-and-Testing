#!/usr/bin/env python3
"""
4h_1d_kama_rsi_chop_v2
KAMA trend + RSI(14) + chop regime filter.
Long when KAMA rising + RSI > 50 + chop > 61.8 (range) for mean reversion to upper band.
Short when KAMA falling + RSI < 50 + chop > 61.8 for mean reversion to lower band.
Exit when RSI crosses 50 or chop < 38.2 (trending).
Uses 1d KAMA for trend, 4h RSI and chop for entry/exit.
Designed for low trade frequency (target: 20-40 trades/year).
Works in both trending and ranging markets by combining trend filter with mean reversion in chop.
"""

name = "4h_1d_kama_rsi_chop_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=er_length, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d KAMA (10,2,30)
    kama_1d = calculate_kama(close_1d, er_length=10, fast=2, slow=30)
    kama_1d_slope = np.diff(kama_1d, prepend=kama_1d[0])
    
    # Align KAMA and slope to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h Choppy Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_slope_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        kama_rising = kama_slope_aligned[i] > 0
        kama_falling = kama_slope_aligned[i] < 0
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        chop_high = chop[i] > 61.8  # ranging market
        chop_low = chop[i] < 38.2   # trending market
        
        # Long entry: KAMA up + RSI > 50 + chop > 61.8 (mean reversion to upper band in range)
        if kama_rising and rsi_above_50 and chop_high and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: KAMA down + RSI < 50 + chop > 61.8 (mean reversion to lower band in range)
        elif kama_falling and rsi_below_50 and chop_high and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (rsi[i] < 50 or chop_low):  # RSI crosses 50 or trend starts
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] > 50 or chop_low):  # RSI crosses 50 or trend starts
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