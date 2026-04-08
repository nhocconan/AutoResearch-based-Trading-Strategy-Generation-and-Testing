#!/usr/bin/env python3
"""
Daily KAMA with RSI Filter and Chop Regime Filter
Hypothesis: KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends. Combined with RSI for momentum and chop filter to avoid ranging markets, this should work in both bull and bear markets with low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter and chop regime
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # KAMA calculation on daily close
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.full_like(close, np.nan, dtype=float)
        kama[length-1] = close[length-1]
        for i in range(length, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # RSI(14) on daily close
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Choppiness Index on weekly data
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = np.zeros_like(close)
        atr[0] = tr[0]
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        highest_high[0] = high[0]
        lowest_low[0] = low[0]
        for i in range(1, len(close)):
            highest_high[i] = max(highest_high[i-1], high[i])
            lowest_low[i] = min(lowest_low[i-1], low[i])
        
        hh_ll = highest_high - lowest_low
        sum_atr = np.zeros_like(close)
        for i in range(period-1, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if hh_ll[i] != 0:
                chop[i] = 100 * np.log10(sum_atr[i] / hh_ll[i]) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_chop(high_1w, low_1w, close_1w, period=14)
    
    # Align weekly indicators to daily
    kama_aligned = kama  # Already daily
    rsi_aligned = rsi    # Already daily
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is trending (CHOP < 38.2) or extreme ranging (CHOP > 61.8 for mean reversion)
        # We'll use trending market for trend following
        is_trending = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR chop enters ranging market
            if (close[i] < kama_aligned[i] or 
                chop_aligned[i] > 50):  # Exit when not strongly trending
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR chop enters ranging market
            if (close[i] > kama_aligned[i] or 
                chop_aligned[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in trending markets
            if is_trending:
                # Long: price above KAMA AND RSI > 50 (bullish momentum)
                if (close[i] > kama_aligned[i] and 
                    rsi_aligned[i] > 50):
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA AND RSI < 50 (bearish momentum)
                elif (close[i] < kama_aligned[i] and 
                      rsi_aligned[i] < 50):
                    position = -1
                    signals[i] = -0.25
    
    return signals