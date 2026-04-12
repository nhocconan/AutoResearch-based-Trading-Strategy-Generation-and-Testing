#!/usr/bin/env python3
"""
6h_12h_ElderRay_Power_Reversal
Hypothesis: Uses 12h Elder Ray (bull/bear power) with 6m zero-cross signals and volume confirmation.
Elder Ray > 0 indicates bull power > 0, < 0 indicates bear power > 0. Zero-cross signals trend changes.
Works in both bull/bear markets by capturing momentum shifts with volume filter to avoid false signals.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ElderRay_Power_Reversal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H ELDER RAY (BULL/BEAR POWER) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 13-period EMA of close for Elder Ray
    def ema(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema_13 = ema(close_12h, 13)
    
    # Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    bull_power = high_12h - ema_13
    bear_power = ema_13 - low_12h
    
    # Align to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        # Long: Bull Power crosses above zero with volume confirmation
        long_signal = (bull_power_6h[i] > 0 and bull_power_6h[i-1] <= 0 and vol_ratio[i] > 1.5)
        # Short: Bear Power crosses above zero with volume confirmation
        short_signal = (bear_power_6h[i] > 0 and bear_power_6h[i-1] <= 0 and vol_ratio[i] > 1.5)
        
        # Exit conditions
        # Exit long when Bear Power crosses above zero (bearish momentum takes over)
        exit_long = (position == 1) and (bear_power_6h[i] > 0 and bear_power_6h[i-1] <= 0)
        # Exit short when Bull Power crosses above zero (bullish momentum takes over)
        exit_short = (position == -1) and (bull_power_6h[i] > 0 and bull_power_6h[i-1] <= 0)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals