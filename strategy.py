#!/usr/bin/env python3

# 1d_1w_kama_rsi_chop_filter_v1
# Hypothesis: Daily KAMA trend + RSI momentum + Choppiness regime filter to avoid whipsaws.
# KAMA adapts to market noise, reducing false signals in ranging markets.
# RSI captures momentum extremes, while Choppiness filter ensures we only trade in clear trends.
# Works in bull/bear by using adaptive trend (KAMA) and avoiding trades in choppy markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_1w_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency ratio
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if volatility[i-er_length:i+1].sum() > 0:
            er[i] = change[i-er_length:i+1].sum() / volatility[i-er_length:i+1].sum()
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, length=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
    avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, length=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            np.abs(high[i] - close[i-1]),
            np.abs(low[i] - close[i-1])
        )
        atr[i] = tr
    
    # Smoothed ATR
    atr_sum = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=length, min_periods=length).max().values
    ll = pd.Series(low).rolling(window=length, min_periods=length).min().values
    
    # Choppiness formula
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(length)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly KAMA for trend direction
    kama_1w = calculate_kama(close_1w, er_length=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily indicators
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, length=14)
    chop = calculate_choppiness(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when market is trending (Choppiness < 38.2)
        trending_market = chop[i] < 38.2
        
        # Long: price above weekly KAMA AND RSI > 50 (bullish momentum)
        if (close[i] > kama_1w_aligned[i] and rsi[i] > 50 and trending_market and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: price below weekly KAMA AND RSI < 50 (bearish momentum)
        elif (close[i] < kama_1w_aligned[i] and rsi[i] < 50 and trending_market and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or RSI crosses 50 in opposite direction
        elif position == 1 and (rsi[i] < 50 or close[i] < kama_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] > 50 or close[i] > kama_1w_aligned[i]):
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