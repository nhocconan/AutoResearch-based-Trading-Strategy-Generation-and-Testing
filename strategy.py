#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_and_Chop_Regime
Hypothesis: KAMA(14) trend direction filtered by RSI(14) extremes and Choppiness Index(14) regime.
Long when KAMA trending up, RSI < 40 (mean-reversion opportunity in uptrend), and choppy market (CHOP > 61.8).
Short when KAMA trending down, RSI > 60, and choppy market.
Uses mean-reversion within trending markets to capture swings in both bull and bear regimes.
Designed for low trade frequency with multiple filters to avoid overtrading.
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
    
    # KAMA calculation
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        # KAMA
        kama = np.full_like(close, np.nan)
        kama[length-1] = close[length-1]
        for i in range(length, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Choppiness Index
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        sum_atr = np.nancumsum(atr)
        hh = np.maximum.accumulate(high)
        ll = np.minimum.accumulate(low)
        range_hl = hh - ll
        chop = 100 * np.log10(sum_atr / range_hl) / np.log10(length)
        return chop
    
    # Calculate indicators
    kama_val = kama(close, 10, 2, 30)
    rsi_input = pd.Series(close)
    delta = rsi_input.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    chop_val = chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama_val[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop_val[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_now = kama_val[i]
        kama_prev = kama_val[i-1]
        rsi_now = rsi[i]
        chop_now = chop_val[i]
        
        # KAMA trend direction
        kama_up = kama_now > kama_prev
        kama_down = kama_now < kama_prev
        
        # Choppy market condition (range-bound)
        choppy = chop_now > 61.8
        
        if position == 0:
            # Long: KAMA up, RSI oversold, choppy market
            if kama_up and rsi_now < 40 and choppy:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, choppy market
            elif kama_down and rsi_now > 60 and choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: KAMA turns down OR RSI overbought
            if kama_down or rsi_now > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI oversold
            if kama_up or rsi_now < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_With_RSI_and_Chop_Regime"
timeframe = "4h"
leverage = 1.0