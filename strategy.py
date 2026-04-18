#!/usr/bin/env python3
"""
4h_KAMA_Trend_with_Volume_and_Chop
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) as trend filter on 4h, combined with volume confirmation and Choppiness Index regime filter. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. Volume confirms institutional participation. Choppiness Index avoids trending markets where mean reversion fails. Targets 20-30 trades/year by requiring KAMA alignment, volume > 1.5x average, and Choppiness > 61.8 (range) for mean-reversion entries or < 38.2 (trend) for trend-following entries. Works in bull markets by following KAMA uptrend with volume, and in bear markets by taking counter-trend reversals only in high-chop conditions.
"""

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
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, kama_period))  # |close - close[kama_period]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of abs changes over kama_period
    
    # Pad arrays to match length
    change = np.concatenate([np.full(kama_period, np.nan), change])
    volatility = np.concatenate([np.full(kama_period, np.nan), volatility])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[kama_period] = close[kama_period]  # seed
    for i in range(kama_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Choppiness Index (14-period)
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr if i == 1 else (atr[i-1] * 13 + tr) / 14  # Wilder smoothing
    
    # Sum of true ranges over 14 periods
    sum_tr = np.full(n, np.nan)
    for i in range(14, n):
        sum_tr[i] = np.sum(atr[i-13:i+1])  # sum of last 14 TR values
    
    # Highest high and lowest low over 14 periods
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum_tr / (max_high - min_low)) / log10(14)
    chop = np.full(n, np.nan)
    range_val = max_high - min_low
    chop = np.where((range_val > 0) & (~np.isnan(sum_tr)), 
                    100 * np.log10(sum_tr / range_val) / np.log10(14), 
                    50)  # default to neutral when range is zero
    
    chop_long_threshold = 61.8  # high chop = range (mean revert)
    chop_short_threshold = 38.2  # low chop = trend (trend follow)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period + 1, 20, 14)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price > KAMA, volume confirmation, and either:
            # 1. High chop (range) -> mean reversion: price < KAMA but reverting up
            # 2. Low chop (trend) -> trend follow: price > KAMA
            if (volume[i] > vol_ma[i] * 1.5):
                if chop[i] > chop_long_threshold:
                    # Mean reversion in range: buy when price dips below KAMA but shows strength
                    if close[i] < kama[i] and close[i] > close[i-1]:
                        signals[i] = 0.25
                        position = 1
                elif chop[i] < chop_short_threshold:
                    # Trend following: buy when price above KAMA in trending market
                    if close[i] > kama[i]:
                        signals[i] = 0.25
                        position = 1
            # Short entry: price < KAMA, volume confirmation, and either:
            # 1. High chop (range) -> mean reversion: price > KAMA but reverting down
            # 2. Low chop (trend) -> trend follow: price < KAMA
            if (volume[i] > vol_ma[i] * 1.5):
                if chop[i] > chop_long_threshold:
                    # Mean reversion in range: sell when price rises above KAMA but shows weakness
                    if close[i] > kama[i] and close[i] < close[i-1]:
                        signals[i] = -0.25
                        position = -1
                elif chop[i] < chop_short_threshold:
                    # Trend following: sell when price below KAMA in trending market
                    if close[i] < kama[i]:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA or loses volume confirmation
            if (close[i] < kama[i] or 
                volume[i] <= vol_ma[i] * 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA or loses volume confirmation
            if (close[i] > kama[i] or 
                volume[i] <= vol_ma[i] * 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_with_Volume_and_Chop"
timeframe = "4h"
leverage = 1.0