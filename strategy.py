#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend + RSI(14) + Choppiness regime filter
    # Long: KAMA trending up + RSI < 40 (pullback in uptrend) + Chop > 61.8 (range)
    # Short: KAMA trending down + RSI > 60 (pullback in downtrend) + Chop > 61.8 (range)
    # Uses mean-reversion in range-bound markets with trend filter to avoid whipsaw
    # Target: 7-25 trades/year to stay within 1d optimal range (30-100 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = KAMA[1] + SC * (Close - KAMA[1])
    # Using fast=2, slow=30 as typical
    change = np.abs(np.diff(close, n=10))  # |Close - Close[10]|
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = np.abs(close[i] - close[i-1])
    
    # Sum of volatility over 10 periods
    vol_sum = np.zeros(n)
    for i in range(10, n):
        vol_sum[i] = np.sum(volatility[i-9:i+1])
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(10, n):
        if vol_sum[i] > 0:
            er[i] = change[i] / vol_sum[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fastest = 2.0 / (2 + 1)  # 2-period EMA
    slowest = 2.0 / (30 + 1)  # 30-period EMA
    sc = np.zeros(n)
    for i in range(10, n):
        sc[i] = (er[i] * (fastest - slowest) + slowest) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if i >= 10:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]  # Initialize with price
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < 14:
            if i > 0:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i-1]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i-1]) / i
            else:
                avg_gain[i] = gain[i-1] if i > 0 else 0
                avg_loss[i] = loss[i-1] if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 50
    
    # Choppiness Index (14-period)
    # Chop = 100 * log10(sum(ATR) / (highest high - lowest low)) / log10(14)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr_sum = np.zeros(n)
    for i in range(14, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    chop = np.zeros(n)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI conditions for mean reversion
        rsi_oversold = rsi[i] < 40  # Pullback in uptrend
        rsi_overbought = rsi[i] > 60  # Pullback in downtrend
        
        # Choppiness regime filter: Chop > 61.8 indicates ranging market
        chop_high = chop[i] > 61.8
        
        # Entry conditions
        long_signal = kama_up and rsi_oversold and chop_high
        short_signal = kama_down and rsi_overbought and chop_high
        
        # Exit conditions: reverse signal or chop low (trending market)
        exit_long = position == 1 and (not kama_up or chop[i] <= 61.8)
        exit_short = position == -1 and (not kama_down or chop[i] <= 61.8)
        
        # Execute signals
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "1d_kama_rsi_chop_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0