#!/usr/bin/env python3
"""
1d_KAMA_Trend_WeeklyTrend_Filter
Hypothesis: Daily KAMA trend with weekly trend filter to avoid whipsaw in choppy markets.
Long when KAMA slope turns up + price > KAMA + weekly close > weekly EMA34.
Short when KAMA slope turns down + price < KAMA + weekly close < weekly EMA34.
Exit on opposite signal. Position size: ±0.25. Designed to work in both bull and bear regimes.
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(1, len(close)):
            if volatility[i-er_length+1:i+1].sum() > 0:
                er[i] = change[i-er_length+1:i+1].sum() / volatility[i-er_length+1:i+1].sum()
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama_vals, prepend=0)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(10, 34)  # KAMA and weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_vals[i]) or 
            np.isnan(kama_slope[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA-based signals
        kama_bullish = kama_slope[i] > 0 and close[i] > kama_vals[i]
        kama_bearish = kama_slope[i] < 0 and close[i] < kama_vals[i]
        
        if position == 0:
            # Long: KAMA bullish + weekly uptrend
            if kama_bullish and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish + weekly downtrend
            elif kama_bearish and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns bearish
            if kama_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns bullish
            if kama_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0