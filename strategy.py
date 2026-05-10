#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing a smooth trend filter.
# Combined with volume confirmation (>1.5x average) and Choppiness Index (CHOP > 61.8 for ranging),
# the strategy enters long when price > KAMA in ranging markets, and short when price < KAMA.
# Exits when price crosses back over KAMA. Designed for low trade frequency (15-25/year)
# to minimize fee drag. Works in ranging markets via mean-reversion at KAMA and in trending
# markets via trend-following when CHOP < 38.2 (trending) but only in direction of KAMA slope.

name = "4h_KAMA_Trend_With_Volume_and_Chop_Filter"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on close
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # Initialize KAMA
        kama_out = np.full_like(close, np.nan)
        kama_out[period] = close[period]
        for i in range(period+1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close, period=10, fast=2, slow=30)
    
    # Calculate Choppiness Index (CHOP) on daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr_period = 14
    atr = np.full_like(close_1d, np.nan)
    if len(close_1d) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(close_1d)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of ATR over period
    chop_period = 14
    sum_atr = np.full_like(close_1d, np.nan)
    if len(close_1d) >= chop_period:
        for i in range(chop_period-1, len(close_1d)):
            sum_atr[i] = np.sum(atr[i-chop_period+1:i+1])
    
    # Max and min range over period
    max_h = np.full_like(close_1d, np.nan)
    min_l = np.full_like(close_1d, np.nan)
    if len(close_1d) >= chop_period:
        for i in range(chop_period-1, len(close_1d)):
            max_h[i] = np.max(high_1d[i-chop_period+1:i+1])
            min_l[i] = np.min(low_1d[i-chop_period+1:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1d, np.nan)
    for i in range(chop_period-1, len(close_1d)):
        if sum_atr[i] > 0 and (max_h[i] - min_l[i]) > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_h[i] - min_l[i])) / np.log10(chop_period)
        else:
            chop[i] = 50  # neutral if undefined
    
    # Align CHOP to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need enough history for KAMA and volume
    
    for i in range(start_idx, n):
        if np.isnan(kama_val[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price above KAMA with volume confirmation
            if close[i] > kama_val[i] and volume_confirm:
                # In ranging markets, mean revert to KAMA (but we enter when above)
                # In trending markets, follow trend (KAMA slope up)
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume confirmation
            elif close[i] < kama_val[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below KAMA
            if close[i] < kama_val[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above KAMA
            if close[i] > kama_val[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals