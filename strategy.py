#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and choppiness regime filter
# Long when price > KAMA AND RSI < 40 AND chop > 61.8 (range market)
# Short when price < KAMA AND RSI > 60 AND chop > 61.8 (range market)
# Exit when price crosses KAMA (trend reversal) OR RSI reaches opposite extreme (60/40)
# Uses 1d primary timeframe with 1w HTF for choppiness regime filter
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) based on proven 1d mean reversion performance
# Works in both bull and bear markets by using choppiness regime to avoid trending markets where mean reversion fails

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data ONCE before loop for choppiness regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close for trend
    # KAMA requires Efficiency Ratio (ER) and smoothing constants
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate Choppiness Index(14) on 1w data
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(14))) / log10(n)
    atr_1w = np.maximum(np.maximum(df_1w['high'] - df_1w['low'],
                                   np.abs(df_1w['high'] - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]]))),
                        np.abs(df_1w['low'] - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]])))
    # Sum of ATR over 1 period (TR) for denominator
    tr_sum_1 = pd.Series(atr_1w).rolling(window=1, min_periods=1).sum().values
    # ATR over 14 periods
    atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).mean().values
    # Chop calculation
    chop = np.full(len(df_1w), np.nan)
    for i in range(13, len(df_1w)):
        if atr_14[i] > 0:
            chop[i] = 100 * np.log10(tr_sum_1[i] / (14 * atr_14[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when no volatility
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > KAMA AND RSI < 40 AND chop > 61.8 (range market)
            if (close[i] > kama[i] and 
                rsi[i] < 40 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < KAMA AND RSI > 60 AND chop > 61.8 (range market)
            elif (close[i] < kama[i] and 
                  rsi[i] > 60 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI reaches 60 (overbought in range)
            if close[i] < kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI reaches 40 (oversold in range)
            if close[i] > kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals