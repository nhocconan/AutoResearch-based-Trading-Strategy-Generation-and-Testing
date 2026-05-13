#!/usr/bin/env python3
"""
6h_KAMA_Trend_Regime_Adaptive
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) to identify trend strength and direction, combined with a regime filter based on price position relative to KAMA and Bollinger Bands to avoid whipsaws. In trending regimes (price outside Bollinger Bands), follow KAMA direction. In ranging regimes (price inside Bollinger Bands), fade extremes at Bollinger Bands. Uses 1d trend filter to avoid counter-trend trades. Designed for 6h timeframe to balance signal frequency and noise reduction, targeting 15-35 trades/year.
"""

name = "6h_KAMA_Trend_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate KAMA (6h timeframe)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix dimensions: volatility needs to be same length as change
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+10]))) for i in range(len(change))])
    # Pad beginning with NaN for first 9 values
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at first valid point
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate Bollinger Bands (20, 2) on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime: trending if price outside Bollinger Bands
        is_trending = close[i] > upper_bb[i] or close[i] < lower_bb[i]
        
        if position == 0:
            # LONG conditions
            if is_trending:
                # In trending regime: follow KAMA direction (price above KAMA)
                if close[i] > kama[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            else:
                # In ranging regime: fade at lower Bollinger Band
                if close[i] <= lower_bb[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT conditions
            if is_trending:
                # In trending regime: follow KAMA direction (price below KAMA)
                if close[i] < kama[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging regime: fade at upper Bollinger Band
                if close[i] >= upper_bb[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: reverse conditions
            if is_trending:
                # Exit when price crosses below KAMA or trend fails
                if close[i] < kama[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit when price moves back above middle or hits upper band
                if close[i] >= sma_20[i] or close[i] >= upper_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: reverse conditions
            if is_trending:
                # Exit when price crosses above KAMA or trend fails
                if close[i] > kama[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit when price moves back below middle or hits lower band
                if close[i] <= sma_20[i] or close[i] <= lower_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals