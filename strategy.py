#!/usr/bin/env python3
"""
6h_1d_Fisher_Transform_Reversal
Hypothesis: Ehlers Fisher Transform on 1d closes identifies extreme reversals. 
Long when Fisher crosses below -1.5 (oversold) with 6h bullish candle.
Short when Fisher crosses above +1.5 (overbought) with 6h bearish candle.
Exit when Fisher crosses back through zero.
Works in both bull/bear markets by capturing mean-reversion swings.
Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
"""

name = "6h_1d_Fisher_Transform_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_ft_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Fisher Transform
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # --- 1d Fisher Transform (Ehlers) ---
    # Normalize price to [-1, 1] using recent min/max
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate median price for smoother input
    median_price = (high_1d + low_1d) / 2.0
    
    # Normalize to [-1, 1] over 10-period lookback
    def normalize_series(series, length):
        highest = pd.Series(series).rolling(window=length, min_periods=length).max().values
        lowest = pd.Series(series).rolling(window=length, min_periods=length).min().values
        # Avoid division by zero
        range_val = highest - lowest
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = 2 * ((series - lowest) / range_val) - 1
        # Clamp to [-0.999, 0.999] to prevent math domain error
        normalized = np.clip(normalized, -0.999, 0.999)
        return normalized
    
    normalized_price = normalize_series(median_price, 10)
    
    # Fisher Transform: 0.5 * ln((1+x)/(1-x))
    fish = 0.5 * np.log((1 + normalized_price) / (1 - normalized_price))
    
    # Smooth with 3-period EMA
    fish_smoothed = pd.Series(fish).ewm(span=3, adjust=False).mean().values
    
    # Align Fisher to 6h timeframe
    fish_aligned = align_htf_to_ltf(prices, df_1d, fish_smoothed)
    
    # --- Entry Conditions ---
    # Bullish 6h candle: close > open
    bullish_candle = close_6h > open_6h
    # Bearish 6h candle: close < open
    bearish_candle = close_6h < open_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for calculations
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if Fisher is not available
        if np.isnan(fish_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        fish_val = fish_aligned[i]
        fish_prev = fish_aligned[i-1] if i > 0 else fish_val
        
        if position == 0:
            # Long: Fisher crosses below -1.5 (oversold) with bullish candle
            if fish_prev > -1.5 and fish_val <= -1.5 and bullish_candle[i]:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses above +1.5 (overbought) with bearish candle
            elif fish_prev < 1.5 and fish_val >= 1.5 and bearish_candle[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when Fisher crosses zero (mean reversion complete)
            if position == 1:
                # Exit long: Fisher crosses above zero
                if fish_prev <= 0 and fish_val > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Fisher crosses below zero
                if fish_prev >= 0 and fish_val < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals