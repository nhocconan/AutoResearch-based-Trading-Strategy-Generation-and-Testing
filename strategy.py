#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Use KAMA(10,2,30) for adaptive trend direction, RSI(14) for momentum confirmation, and Choppiness Index(14) to avoid ranging markets. Enter long when KAMA up, RSI > 50, and CHOP < 38.2 (trending); enter short when KAMA down, RSI < 50, and CHOP < 38.2. Exit on opposite signal. Discrete sizing: 0.25. Target: 30-60 trades/year to balance edge and fee drag.
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
    
    # KAMA calculation (adaptive moving average)
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # not quite right, need rolling sum
        # Correct volatility: rolling sum of absolute changes
        volatility = pd.Series(close).rolling(window=er_length, min_periods=1).apply(
            lambda x: np.sum(np.abs(np.diff(x))), raw=True
        ).values
        # Avoid division by zero
        er = np.where(volatility > 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Choppiness Index
    def chop(high, low, close, window=14):
        atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=window, min_periods=window).mean()
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(window)
        return chop.values
    
    # RSI
    def rsi(close, window=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=window, adjust=False, min_periods=window).mean()
        avg_loss = pd.Series(loss).ewm(span=window, adjust=False, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    # Pre-compute indicators
    kama_val = kama(close, er_length=10, fast_sc=2, slow_sc=30)
    chop_val = chop(high, low, close, window=14)
    rsi_val = rsi(close, window=14)
    
    # Trend: KAMA slope (1-bar change)
    kama_slope = kama_val - np.roll(kama_val, 1)
    kama_slope[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 14  # need RSI, CHOP, KAMA slope
    
    for i in range(start_idx, n):
        # Skip if any indicator not ready
        if (np.isnan(kama_val[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(chop_val[i]) or np.isnan(rsi_val[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Conditions
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        rsi_above_50 = rsi_val[i] > 50
        rsi_below_50 = rsi_val[i] < 50
        chop_low = chop_val[i] < 38.2  # trending market
        
        if position == 0:
            # Long: KAMA up, RSI > 50, trending market
            if kama_up and rsi_above_50 and chop_low:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, trending market
            elif kama_down and rsi_below_50 and chop_low:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold
            signals[i] = 0.25
            # Exit: KAMA down OR RSI < 40 OR choppy market (CHOP > 61.8)
            if kama_down or rsi_val[i] < 40 or chop_val[i] > 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold
            signals[i] = -0.25
            # Exit: KAMA up OR RSI > 60 OR choppy market (CHOP > 61.8)
            if kama_up or rsi_val[i] > 60 or chop_val[i] > 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0