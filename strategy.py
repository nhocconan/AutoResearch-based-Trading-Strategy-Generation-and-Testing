#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_filter_v1
Hypothesis: Daily strategy using KAMA for trend direction, RSI for momentum, and Choppiness Index for regime filtering.
Trades only when KAMA trend aligns with RSI momentum in appropriate chop regime.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag and work in both bull/bear markets.
"""

name = "1d_1w_kama_rsi_chop_filter_v1"
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
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    def calculate_kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[length] = close[length]
        for i in range(length+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Choppiness Index - regime filter
    def calculate_choppiness(high, low, close, length=14):
        atr = np.zeros(len(close))
        atr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = (atr[i-1] * (length-1) + tr) / length if i < length else (atr[i-1] * (length-1) + tr) / length
        
        sum_atr = np.sum(atr[-length:]) if len(atr) >= length else np.sum(atr)
        highest_high = np.max(high[-length:]) if len(high) >= length else np.max(high)
        lowest_low = np.min(low[-length:]) if len(low) >= length else np.min(low)
        range_max_min = highest_high - lowest_low
        
        if range_max_min > 0 and sum_atr > 0:
            chop = 100 * np.log10(sum_atr / range_max_min) / np.log10(length)
        else:
            chop = 50
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    rsi = np.full_like(close, np.nan, dtype=float)
    if len(close) >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Choppiness Index (daily)
    chop = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        chop[i] = calculate_choppiness(high[max(0,i-13):i+1], low[max(0,i-13):i+1], close[max(0,i-13):i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price > KAMA (uptrend), RSI > 50 (bullish momentum), chop < 61.8 (trending market), price > weekly EMA20
        if (close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8 and close[i] > ema20_1w_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: price < KAMA (downtrend), RSI < 50 (bearish momentum), chop < 61.8 (trending market), price < weekly EMA20
        elif (close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8 and close[i] < ema20_1w_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or chop > 61.8 (choppy market)
        elif position == 1 and (close[i] < kama[i] or chop[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama[i] or chop[i] > 61.8):
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