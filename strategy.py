#!/usr/bin/env python3
# 12h_KAMA_1wTrend_VolumeFilter
# Hypothesis: 12h strategy using KAMA trend filter with weekly trend and volume confirmation.
# Enters long when KAMA turns upward (bullish slope), price above KAMA, weekly trend up, and volume > 1.5x average.
# Enters short when KAMA turns downward (bearish slope), price below KAMA, weekly trend down, and volume > 1.5x average.
# Exits when price crosses back below/above KAMA.
# Designed for low trade frequency (<25/year) with strong trend following in both bull and bear markets.
# Uses weekly trend filter to avoid counter-trend trades, reducing whipsaw in choppy markets.

name = "12h_KAMA_1wTrend_VolumeFilter"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # This is incorrect, let's fix properly
    
    # Correct KAMA calculation
    dir = np.abs(np.diff(close, prepend=close[0]))  # direction = |close - close_prev|
    vol = np.abs(np.diff(close, prepend=close[0]))  # volatility = sum of |close - close_prev| over window
    
    # We'll compute over a 10-period window for ER
    window = 10
    dir_sum = pd.Series(dir).rolling(window=window, min_periods=1).sum().values
    vol_sum = pd.Series(vol).rolling(window=window, min_periods=1).sum().values
    
    # Avoid division by zero
    er = np.where(vol_sum > 0, dir_sum / vol_sum, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate KAMA slope (trend direction) - positive = upward slope
    kama_slope = kama - np.roll(kama, 1)
    kama_slope[0] = 0
    
    # Weekly trend: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_slope = ema34_1w - np.roll(ema34_1w, 1)
    ema34_1w_slope[0] = 0
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema34_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w_slope)
    
    # Volume filter: 1.5x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_1w_slope_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA slope up, price above KAMA, weekly trend up, volume spike
            if (kama_slope[i] > 0 and 
                close[i] > kama[i] and 
                ema34_1w_slope_aligned[i] > 0 and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA slope down, price below KAMA, weekly trend down, volume spike
            elif (kama_slope[i] < 0 and 
                  close[i] < kama[i] and 
                  ema34_1w_slope_aligned[i] < 0 and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals