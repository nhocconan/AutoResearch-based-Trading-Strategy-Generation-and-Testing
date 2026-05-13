#!/usr/bin/env python3
"""
6h_Liquidity_Sweep_And_Reversal
Hypothesis: Price often sweeps liquidity (equal highs/lows) before reversing. 
This strategy identifies liquidity sweeps using equal highs/lows with volume 
confirmation, then enters on reversal. Works in both bull and bear markets by 
trapping stops and capitalizing on reversal moves. Uses 1d trend filter to 
avoid counter-trend trades.
"""

name = "6h_Liquidity_Sweep_And_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def find_equal_levels(arr, lookback, tolerance_pct=0.001):
    """Find equal highs/lows within tolerance percentage"""
    n = len(arr)
    equal_high = np.zeros(n, dtype=bool)
    equal_low = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        window_high = arr[i-lookback:i]
        window_low = arr[i-lookback:i]
        
        # Check for equal high (current high matches any in lookback)
        if np.any(np.abs(arr[i] - window_high) / window_high <= tolerance_pct):
            equal_high[i] = True
        # Check for equal low (current low matches any in lookback)
        if np.any(np.abs(arr[i] - window_low) / window_low <= tolerance_pct):
            equal_low[i] = True
            
    return equal_high, equal_low

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 50-period EMA on daily for trend filter
    daily_close_series = pd.Series(daily_close)
    ema_50_1d = daily_close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Find liquidity sweeps (equal highs/lows) with 20-bar lookback
    equal_high, equal_low = find_equal_levels(high, 20, 0.001)
    equal_low_high, equal_high_low = find_equal_levels(low, 20, 0.001)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LIQUIDITY SWEEP LONG: Sweep lows with volume, then reverse
            # Condition: Equal low touched + volume + price above daily EMA50 (uptrend bias)
            if equal_low[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]:
                # Additional reversal confirmation: close > open (bullish candle)
                if close[i] > prices['open'].iloc[i]:
                    signals[i] = 0.25
                    position = 1
            # LIQUIDITY SWEEP SHORT: Sweep highs with volume, then reverse
            # Condition: Equal high touched + volume + price below daily EMA50 (downtrend bias)
            elif equal_high[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]:
                # Additional reversal confirmation: close < open (bearish candle)
                if close[i] < prices['open'].iloc[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches equal high (liquidity) or shows weakness
            if equal_high[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches equal low (liquidity) or shows strength
            if equal_low[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals