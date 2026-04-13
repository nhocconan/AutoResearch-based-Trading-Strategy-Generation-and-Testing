#!/usr/bin/env python3
"""
12h_1d_KAMA_Trend_With_Regime_Filter
Hypothesis: Use 1d KAMA direction as primary trend filter (adaptive to market conditions) and enter on 12h pullbacks to KAMA with volume confirmation. 
Avoids whipsaw by using Choppiness Index regime filter: only trade when market is trending (CHOP < 38.2). 
Designed for 12h timeframe to keep trades low (target: 20-40/year) and reduce fee drag. 
Works in both bull (follows trend) and bear (avoids false signals in range) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[29] = close_1d[29]  # start after enough data
    for i in range(30, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate Choppiness Index on 1d data (14-period)
    def calculate_choppiness(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       np.abs(high_arr[i] - close_arr[i-1]),
                       np.abs(low_arr[i] - close_arr[i-1]))
        # Smooth TR with Wilder's smoothing (equivalent to RMA)
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Chop calculation
        sum_tr = np.zeros_like(close_arr)
        for i in range(period, len(tr)+1):
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
        max_high = np.zeros_like(close_arr)
        min_low = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            max_high[i] = np.max(high_arr[i-period+1:i+1])
            min_low[i] = np.min(low_arr[i-period+1:i+1])
        chop = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            if max_high[i] - min_low[i] != 0:
                chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    chop = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Align 1d data to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        regime_filter = chop_aligned[i] < 38.2
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Trend condition: price relative to KAMA
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # Entry conditions with pullback logic
        if position == 0:
            # Long: price above KAMA (uptrend) and pulling back to touch or slightly below KAMA
            if price_above_kama and close[i] <= kama_aligned[i] * 1.005 and regime_filter and vol_condition:
                position = 1
                signals[i] = position_size
            # Short: price below KAMA (downtrend) and pulling back to touch or slightly above KAMA
            elif price_below_kama and close[i] >= kama_aligned[i] * 0.995 and regime_filter and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price crosses below KAMA or regime changes to choppy
            if close[i] < kama_aligned[i] * 0.995 or chop_aligned[i] >= 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when price crosses above KAMA or regime changes to choppy
            if close[i] > kama_aligned[i] * 1.005 or chop_aligned[i] >= 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_Trend_With_Regime_Filter"
timeframe = "12h"
leverage = 1.0