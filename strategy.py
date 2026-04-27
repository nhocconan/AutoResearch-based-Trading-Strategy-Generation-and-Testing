#!/usr/bin/env python3
"""
#100950 - 1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: 1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for momentum and Choppiness Index for regime filtering.
Trades only when KAMA indicates trend, RSI confirms momentum, and market is trending (CHOP < 38.2).
Designed for low trade frequency (7-25/year) to minimize fee drag. Works in bull (trend following)
and bear (avoids ranging markets via chop filter).
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate KAMA (1d)
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / volatility[length-1:]
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # Calculate RSI (14)
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    # Calculate Choppiness Index (14)
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        # True Range sum
        tr_sum = np.zeros_like(close)
        for i in range(length, len(close)):
            tr_sum[i] = np.sum(atr[i-length+1:i+1])
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(length-1, len(close)):
            hh[i] = np.max(high[i-length+1:i+1])
            ll[i] = np.min(low[i-length+1:i+1])
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if hh[i] - ll[i] != 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_chop(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above KAMA, above weekly EMA50, RSI > 50, Chop < 38.2 (trending)
        if (close[i] > kama[i] and 
            close[i] > ema50_1w_aligned[i] and 
            rsi[i] > 50 and 
            chop[i] < 38.2):
            signals[i] = 0.25
            position = 1
        # Short condition: price below KAMA, below weekly EMA50, RSI < 50, Chop < 38.2 (trending)
        elif (close[i] < kama[i] and 
              close[i] < ema50_1w_aligned[i] and 
              rsi[i] < 50 and 
              chop[i] < 38.2):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal
        elif position == 1 and (close[i] < kama[i] or chop[i] >= 38.2):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > kama[i] or chop[i] >= 38.2):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0