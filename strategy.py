#!/usr/bin/env python3
# Strategy: 6h_1d_AwesomeOscillator_ZeroCross_TrendFilter
# Hypothesis: Awesome Oscillator zero-cross signals filtered by 1d EMA50 trend on 6h timeframe.
# Works in bull markets (buy on bullish AO cross above zero in uptrend) and bear markets (sell on bearish AO cross below zero in downtrend).
# Uses 1d trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Awesome Oscillator on 6h data
    # AO = SMA(median, 5) - SMA(median, 34)
    median_price = (prices['high'].values + prices['low'].values) / 2
    sma5 = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    sma34 = pd.Series(median_price).rolling(window=34, min_periods=34).mean().values
    ao = sma5 - sma34
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ao[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long: AO crosses above zero AND price above 1d EMA50 (uptrend)
            if ao[i] > 0 and ao[i-1] <= 0 and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: AO crosses below zero AND price below 1d EMA50 (downtrend)
            elif ao[i] < 0 and ao[i-1] >= 0 and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: AO crosses below zero or price below 1d EMA50
            if ao[i] < 0 and ao[i-1] >= 0 or price < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: AO crosses above zero or price above 1d EMA50
            if ao[i] > 0 and ao[i-1] <= 0 or price > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_AwesomeOscillator_ZeroCross_TrendFilter"
timeframe = "6h"
leverage = 1.0