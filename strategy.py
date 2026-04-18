#!/usr/bin/env python3
"""
12h_KAMA_Trend_with_Volume_and_Chop
Hypothesis: KAMA adapts to market conditions - in trending markets it tracks price closely, 
in ranging markets it stays flat. Combined with volume confirmation and Choppiness Index 
to identify regime, this should capture trends while avoiding whipsaws. 
Uses 12h primary timeframe with 1w trend filter for multi-timeframe alignment.
Targets 15-25 trades/year by requiring KAMA trend alignment, volume > 2x average, 
and Choppiness < 50 (trending regime). Works in bull markets by following upward KAMA 
adaptation, and in bear markets by taking short positions when KAMA turns down.
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
    er_length = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio and Smoothing Constants
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros(n)
    er[er_length:] = change[er_length-1:] / volatility[er_length-1:]
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly
    ema_20_1w = np.full_like(close_1w, np.nan)
    for i in range(20, len(close_1w)):
        ema_20_1w[i] = np.mean(close_1w[i-20:i]) if i >= 20 else close_1w[i]
    
    # Align weekly EMA to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Choppiness Index (14-period)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period:i])
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(atr_period, n):
        highest_high[i] = np.max(high[i-atr_period:i])
        lowest_low[i] = np.min(low[i-atr_period:i])
    
    chop = np.full(n, 50.0)
    for i in range(atr_period, n):
        if highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i-atr_period:i]) / (highest_high[i] - lowest_low[i])) / np.log10(atr_period)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_length, 20, atr_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above KAMA, weekly EMA uptrend, low chop, volume confirmation
            if (close[i] > kama[i] and close[i] > ema_20_1w_aligned[i] and 
                chop[i] < 50 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA, weekly EMA downtrend, low chop, volume confirmation
            elif (close[i] < kama[i] and close[i] < ema_20_1w_aligned[i] and 
                  chop[i] < 50 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below KAMA or chop increases (ranging market)
            if (close[i] < kama[i] or chop[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA or chop increases (ranging market)
            if (close[i] > kama[i] or chop[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_with_Volume_and_Chop"
timeframe = "12h"
leverage = 1.0