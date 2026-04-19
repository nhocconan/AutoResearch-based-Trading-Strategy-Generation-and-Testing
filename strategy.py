#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA with RSI filter and weekly trend filter.
# Long when: KAMA turns upward, RSI(14) > 50, and weekly EMA21 upward
# Short when: KAMA turns downward, RSI(14) < 50, and weekly EMA21 downward
# Exit when: KAMA reverses direction
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# RSI filters for momentum alignment. Weekly EMA21 ensures trend alignment.
# Target: 15-25 trades/year per symbol. Works in bull (buy strength) and bear (sell weakness).
name = "1d_KAMA_RSI_WeeklyEMA21Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Seed value
    
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rsi = np.zeros_like(close)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    rsi[avg_gain == 0] = 0
    
    # Weekly data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA21 on weekly data
    ema21_1w = np.full_like(close_1w, np.nan)
    ema21_1w[20] = np.mean(close_1w[0:21])
    for i in range(21, len(close_1w)):
        ema21_1w[i] = close_1w[i] * (2/(21+1)) + ema21_1w[i-1] * (1 - 2/(21+1))
    
    # Align weekly EMA21 to daily timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for KAMA and RSI calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_now = rsi[i]
        ema21 = ema21_1w_aligned[i]
        ema21_prev = ema21_1w_aligned[i-1]
        
        if position == 0:
            # Long entry: KAMA turns upward, RSI > 50, weekly EMA21 upward
            if (kama_now > kama_prev and 
                rsi_now > 50 and 
                ema21 > ema21_prev):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA turns downward, RSI < 50, weekly EMA21 downward
            elif (kama_now < kama_prev and 
                  rsi_now < 50 and 
                  ema21 < ema21_prev):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns downward
            if kama_now < kama_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns upward
            if kama_now > kama_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals