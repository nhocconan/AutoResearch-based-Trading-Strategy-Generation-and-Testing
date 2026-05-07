#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter_v4
# Hypothesis: Daily KAMA trend with RSI mean-reversion and Choppiness index regime filter.
# KAMA adapts to market noise - effective in both trending and ranging markets.
# RSI provides mean-reversion signals within the trend context.
# Choppiness index filters for ranging markets (CHOP > 61.8) where mean reversion works best.
# Designed for low trade frequency (10-25/year) with high win rate in ranging markets.

name = "1d_KAMA_RSI_Chop_Filter_v4"
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
    
    # Get weekly data for trend filter and chop calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on weekly close
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[29] = close_1w[29]  # seed
    for i in range(30, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # Align KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align with index
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    # Chop calculation
    chop = np.where((hh - ll) != 0, 100 * np.log10(np.sum(atr, axis=1) / (hh - ll)) / np.log10(14), 50)
    chop = np.concatenate([[np.nan] * 13, chop[13:]])  # align with index
    chop_1w = chop
    
    # Align Chop to daily timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume spike detection: 1.5x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Ensure we have KAMA, RSI, Chop, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend), RSI oversold, choppy market (mean reversion regime)
            if (close[i] < kama_1w_aligned[i] and 
                rsi[i] < 30 and 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (pullback in downtrend), RSI overbought, choppy market
            elif (close[i] > kama_1w_aligned[i] and 
                  rsi[i] > 70 and 
                  chop_1w_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses above KAMA or RSI overbought
            if (close[i] > kama_1w_aligned[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses below KAMA or RSI oversold
            if (close[i] < kama_1w_aligned[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals