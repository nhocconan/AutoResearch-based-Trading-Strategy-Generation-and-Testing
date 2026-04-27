#!/usr/bin/env python3
"""
#100849 - 4h_KAMA_Direction_1dVolatility_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets. 
Combined with 1d volatility filter (low volatility = trend continuation) to avoid false signals during high noise.
Targets 20-30 trades/year to minimize fee drag while maintaining edge in BTC/ETH.
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
    
    # Get 1d data for volatility filter (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # First TR is undefined
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), 
                           axis=0 if len(change.shape) == 0 else None, keepdims=True)
        if len(change.shape) == 0:
            volatility = pd.Series(volatility).rolling(window=er_length, min_periods=1).sum().values
        else:
            volatility = pd.Series(volatility.flatten()).rolling(window=er_length, min_periods=1).sum().values
        
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA on 4h close
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Align 1d ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volatility filter: low volatility environment (ATR below 20-period median)
    atr_median = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).median().values
    low_volatility = atr_14_aligned < (atr_median * 1.2)  # Only trade in relatively low volatility
    
    # Trend direction: price above/below KAMA
    price_above_kama = close > kama
    price_below_kama = close < kama
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_median[i]) or np.isnan(price_above_kama[i]) or
            np.isnan(price_below_kama[i])):
            signals[i] = 0.0
            continue
        
        # Enter long: price above KAMA AND low volatility environment
        if price_above_kama[i] and low_volatility[i]:
            signals[i] = 0.25
            position = 1
        # Enter short: price below KAMA AND low volatility environment
        elif price_below_kama[i] and low_volatility[i]:
            signals[i] = -0.25
            position = -1
        # Exit: volatility expands significantly (potential trend exhaustion or chop)
        elif position == 1 and not low_volatility[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not low_volatility[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Direction_1dVolatility_Filter"
timeframe = "4h"
leverage = 1.0