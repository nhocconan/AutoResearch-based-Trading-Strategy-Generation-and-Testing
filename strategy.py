#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + Chop filter
# Long when KAMA is rising (bullish trend) AND RSI < 30 (oversold) AND Chop > 61.8 (range)
# Short when KAMA is falling (bearish trend) AND RSI > 70 (overbought) AND Chop > 61.8 (range)
# Uses KAMA for trend direction, RSI for mean reversion in extremes, and Chop to filter for ranging markets.
# Designed for low trade frequency (target: 10-25/year) to minimize fee drag and work in both bull and bear markets via mean reversion in ranges.
name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # start at index 9 (10th element)
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    # Handle first element
    kama_rising[0] = False
    kama_falling[0] = False
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Use Wilder's smoothing (alpha = 1/14)
    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)
    avg_gain[13] = np.mean(gain[1:14])  # average of first 13 gains (indices 1-13)
    avg_loss[13] = np.mean(loss[1:14])  # average of first 13 losses
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # For first 14 periods, RSI is undefined
    rsi[:14] = np.nan
    
    # Chop (Choppiness Index) - 14-period
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]  # first period
    # Sum of TR over 14 periods
    sum_tr = np.convolve(tr, np.ones(14), 'valid')
    sum_tr = np.pad(sum_tr, (13, 0), mode='constant', constant_values=np.nan)
    # Highest high and lowest low over 14 periods
    highest_high = np.array([np.max(high[i-13:i+1]) if i >= 13 else np.nan for i in range(n)])
    lowest_low = np.array([np.min(low[i-13:i+1]) if i >= 13 else np.nan for i in range(n)])
    # Chop = 100 * log10(sum_tr / (highest_high - lowest_low)) / log10(14)
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_tr) / np.log10(14), 50)
    chop = np.where(range_hl == 0, 50, chop)
    # For first 13 periods, Chop is undefined
    chop[:13] = np.nan
    
    # Chop > 61.8 indicates ranging market (mean reversion zone)
    chop_range = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for KAMA, RSI, Chop
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or np.isnan(chop_range[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising + RSI < 30 (oversold) + Chop > 61.8 (range)
            long_condition = kama_rising[i] and (rsi[i] < 30) and chop_range[i]
            # Short: KAMA falling + RSI > 70 (overbought) + Chop > 61.8 (range)
            short_condition = kama_falling[i] and (rsi[i] > 70) and chop_range[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA falling or RSI > 50 (momentum) or Chop < 38.2 (trending)
            if kama_falling[i] or (rsi[i] > 50) or (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA rising or RSI < 50 (momentum) or Chop < 38.2 (trending)
            if kama_rising[i] or (rsi[i] < 50) or (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals