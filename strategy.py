#!/usr/bin/env python3
# 6H_1D_Stochastic_Trend_Follow
# Hypothesis: In strong daily trends, stochastic oscillator identifies pullback entries.
# Long when daily trend is up (close > EMA50) and stochastic %K crosses above 20 (oversold bounce).
# Short when daily trend is down (close < EMA50) and stochastic %K crosses below 80 (overbought rejection).
# Uses 1d stochastic(14,3,3) for entry timing and 1d EMA50 for trend filter.
# Works in bull/bear by following daily trend direction. Target: 15-25 trades/year per symbol.

name = "6H_1D_Stochastic_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Stochastic(14,3,3)
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    k_raw = 100 * (close_1d - lowest_low) / (highest_high - lowest_low)
    k_raw = np.where((highest_high - lowest_low) == 0, 50, k_raw)  # avoid division by zero
    k = pd.Series(k_raw).ewm(span=3, adjust=False, min_periods=3).mean().values
    d = pd.Series(k).ewm(span=3, adjust=False, min_periods=3).mean().values  # %D line
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align to 6h
    k_aligned = align_htf_to_ltf(prices, df_1d, k)
    d_aligned = align_htf_to_ltf(prices, df_1d, d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(k_aligned[i]) or np.isnan(d_aligned[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + stochastic crosses above 20 (oversold bounce)
            if bullish and k_aligned[i] > 20 and k_aligned[i-1] <= 20:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + stochastic crosses below 80 (overbought rejection)
            elif bearish and k_aligned[i] < 80 and k_aligned[i-1] >= 80:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or stochastic crosses below 50 (momentum loss)
            if bearish or (k_aligned[i] < 50 and k_aligned[i-1] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or stochastic crosses above 50 (momentum loss)
            if bullish or (k_aligned[i] > 50 and k_aligned[i-1] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals