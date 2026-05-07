#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Reversal_v3"
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
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # will fix below
    
    # Proper volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)  # KAMA is already on 1d? Wait, no: we want KAMA on 1d, but using 1w trend filter.
    # Actually, let's compute KAMA on 1d close, and use 1w EMA20 as trend filter.
    # Recompute KAMA on 1d close price
    change = np.abs(close - np.roll(close, 10))
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first value to match length
    rsi = np.concatenate([[np.nan], rsi])
    
    # Choppy Index (Chop) - using 14-period
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid
    chop = np.where((highest_high - lowest_low) != 0, chop, 50)
    
    # Align 1w EMA20 trend filter
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above KAMA, RSI > 50, Chop > 61.8 (ranging), and 1w uptrend
            if close[i] > kama[i] and rsi[i] > 50 and chop[i] > 61.8 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below KAMA, RSI < 50, Chop > 61.8 (ranging), and 1w downtrend
            elif close[i] < kama[i] and rsi[i] < 50 and chop[i] > 61.8 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below KAMA or RSI < 40 or Chop < 38.2 (trending) or trend turns down
            if close[i] < kama[i] or rsi[i] < 40 or chop[i] < 38.2 or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or RSI > 60 or Chop < 38.2 (trending) or trend turns up
            if close[i] > kama[i] or rsi[i] > 60 or chop[i] < 38.2 or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA (adaptive trend) + RSI momentum + Choppy Index regime filter on 1d.
# Long when price crosses above KAMA (adaptive trend) with RSI > 50 (bullish momentum),
# Chop > 61.8 indicates ranging market (mean reversion favorable), and 1w uptrend (close > 1w EMA20).
# Short when price crosses below KAMA with RSI < 50, Chop > 61.8, and 1w downtrend.
# Exits on KAMA cross in opposite direction, RSI extremes, Chop < 38.2 (trending regime), or 1w trend failure.
# Uses discrete position size (0.25) to minimize churn. Target 15-25 trades/year.
# Works in ranging markets (Chop > 61.8) where mean reversion is effective, with 1w trend filter to avoid counter-trend trades.