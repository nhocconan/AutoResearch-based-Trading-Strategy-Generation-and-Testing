#!/usr/bin/env python3
"""
6h_1w_1d_engulfing_confluence
Uses weekly trend from SMA(50) on 1w and daily engulfing candle patterns on 1d for entry.
Long when weekly uptrend + bullish engulfing on daily, short when weekly downtrend + bearish engulfing on daily.
Exit when weekly trend reverses or opposite engulfing forms.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee flood.
Works in both trending and ranging markets by combining trend filter with price action signals.
"""

name = "6h_1w_1d_engulfing_confluence"
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
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly SMA(50) for trend
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    weekly_uptrend = close_1w > sma_50_1w
    weekly_downtrend = close_1w < sma_50_1w
    
    # Get daily data for engulfing patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bullish engulfing: current bullish candle engulfs previous bearish candle
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d.shift(1)) & (close_1d > open_1d.shift(1)) & (close_1d.shift(1) < open_1d.shift(1))
    # Bearish engulfing: current bearish candle engulfs previous bullish candle
    bearish_engulf = (close_1d < open_1d) & (open_1d > close_1d.shift(1)) & (close_1d < open_1d.shift(1)) & (close_1d.shift(1) > open_1d.shift(1))
    
    # Align weekly trend and daily patterns to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulf)
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulf)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(bullish_engulf_aligned[i]) or np.isnan(bearish_engulf_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: weekly uptrend + bullish engulfing on daily
        if weekly_uptrend_aligned[i] and bullish_engulf_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: weekly downtrend + bearish engulfing on daily
        elif weekly_downtrend_aligned[i] and bearish_engulf_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: weekly trend reverses or opposite engulfing forms
        elif position == 1 and (not weekly_uptrend_aligned[i] or bearish_engulf_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not weekly_downtrend_aligned[i] or bullish_engulf_aligned[i]):
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