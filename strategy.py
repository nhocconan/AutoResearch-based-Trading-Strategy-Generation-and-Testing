#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI + chop filter
# Uses KAMA to determine trend direction, RSI for mean-reversion entry, and Choppiness Index to filter ranging markets.
# Long when: KAMA rising, RSI < 30 (oversold), Chop > 61.8 (ranging)
# Short when: KAMA falling, RSI > 70 (overbought), Chop > 61.8 (ranging)
# Exit when: RSI crosses back to neutral (40-60) or trend changes
# Position size: 0.25 to limit drawdown. Target: 10-25 trades/year.
# Designed to work in both bull (trend continuation) and bear (mean-reversion in ranges) markets.

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0

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
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on weekly close
    close_1w = df_1w['close'].values
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=1)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, change[0]) if len(change) > 0 else np.array([]), change])
    volatility = np.concatenate([np.full(10, volatility[0]) if len(volatility) > 0 else np.array([]), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.0645 - 0.0625) + 0.0625) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    kama_rising = kama > kama_prev
    kama_falling = kama < kama_prev
    kama_rising_aligned = align_htf_to_ltf(prices, df_1w, kama_rising)
    kama_falling_aligned = align_htf_to_ltf(prices, df_1w, kama_falling)
    
    # RSI(14) on daily close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr, axis=1) / (max_high - min_low)) / np.log10(14)
    # Handle first 14 values
    chop[:14] = 50  # neutral value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI < 30, Chop > 61.8 (ranging market)
            if (kama_rising_aligned[i] and 
                rsi[i] < 30 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI > 70, Chop > 61.8 (ranging market)
            elif (kama_falling_aligned[i] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 40 or trend turns down
            if (rsi[i] > 40) or (not kama_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 60 or trend turns up
            if (rsi[i] < 60) or (not kama_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals