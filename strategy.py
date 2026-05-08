#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_Ratio_Trend_Follow_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Previous day's pivot points (HLC/3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Pivot support/resistance levels (R1/S1)
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # === 1d ATR for volatility regime ===
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30_1d = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Align to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    atr10_1d_4h = align_htf_to_ltf(prices, df_1d, atr10_1d)
    atr30_1d_4h = align_htf_to_ltf(prices, df_1d, atr30_1d)
    
    # === 4h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Regime: trending if ATR(10) > ATR(30) * 1.15 ===
    atr_ratio = atr10_1d_4h / (atr30_1d_4h + 1e-10)
    is_trending = atr_ratio > 1.15
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for ATR30
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if is_trending[i]:
                # Trending regime: breakout continuation
                long_cond = (close[i] > r1_4h[i] and 
                            volume[i] > vol_ma20[i])
                short_cond = (close[i] < s1_4h[i] and 
                             volume[i] > vol_ma20[i])
            else:
                # Ranging regime: mean reversion at S1/R1
                long_cond = (close[i] < s1_4h[i] and 
                            volume[i] > vol_ma20[i])
                short_cond = (close[i] > r1_4h[i] and 
                             volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit conditions
            if is_trending[i]:
                # In trending market, exit on breakdown below S1
                exit_cond = close[i] < s1_4h[i]
            else:
                # In ranging market, exit at R1 (mean reversion target)
                exit_cond = close[i] > r1_4h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if is_trending[i]:
                # In trending market, exit on breakout above R1
                exit_cond = close[i] > r1_4h[i]
            else:
                # In ranging market, exit at S1 (mean reversion target)
                exit_cond = close[i] < s1_4h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h strategy using 1d pivot levels (R1/S1) with adaptive regime filtering.
# In trending markets (ATR10 > ATR30 * 1.15): breakout continuation at R1/S1 with volume confirmation.
# In ranging markets: mean reversion at S1/R1 with volume confirmation.
# Uses ATR ratio for regime detection to avoid whipsaws in low volatility.
# Designed for 25-40 trades/year (100-160 over 4 years) to minimize fee drag.
# Works in both bull (trend following) and bear (mean reversion in ranges) markets.