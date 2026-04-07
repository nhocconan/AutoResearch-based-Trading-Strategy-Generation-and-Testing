#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily KAMA + RSI + Choppiness Filter
# Hypothesis: KAMA adapts to market regime (trending/ranging), RSI provides mean reversion signals within trend,
# and Choppiness index filters for ranging conditions. Works in bull via KAMA trend + RSI pullbacks,
# in bear via KAMA trend + RSI bounces. Target: 25-40 trades/year to minimize fee drag.
name = "4h_kama_rsi_chop_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for KAMA and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[0] = 0
    for i in range(1, len(close_1d)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate Choppiness Index on daily data
    atr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                 abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                 abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
        atr_1d[i] = tr
    # Smooth ATR
    atr_smoothed = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    # Calculate highest high and lowest low over 14 periods
    hh = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    # Chop calculation
    chop = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        if atr_smoothed[i] > 0 and (hh[i] - ll[i]) > 0:
            chop[i] = 100 * np.log10(sum(atr_1d[i-13:i+1]) / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate RSI on 4h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Choppiness filter: only trade when market is ranging (Chop > 61.8)
        if chop_aligned[i] <= 61.8:
            # In trending markets, reduce position or stay flat
            if position == 1:
                signals[i] = 0.15  # reduce long
            elif position == -1:
                signals[i] = -0.15  # reduce short
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 70 (overbought) or price crosses below KAMA
            if rsi[i] >= 70 or close[i] < kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: RSI crosses below 30 (oversold) or price crosses above KAMA
            if rsi[i] <= 30 or close[i] > kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price crosses above KAMA AND RSI < 40 (pullback in uptrend)
            if close[i] > kama_aligned[i] and rsi[i] < 40:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below KAMA AND RSI > 60 (bounce in downtrend)
            elif close[i] < kama_aligned[i] and rsi[i] > 60:
                position = -1
                signals[i] = -0.25
    
    return signals