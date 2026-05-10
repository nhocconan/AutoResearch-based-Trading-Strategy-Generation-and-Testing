#!/usr/bin/env python3
"""
12H_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market conditions, capturing trends while avoiding whipsaws in choppy markets.
Combined with RSI(14) for momentum confirmation and volume filter to ensure participation.
Designed for 12h timeframe to capture multi-day trends with low trade frequency (target: 15-30 trades/year).
Works in both bull and bear markets by following KAMA direction, avoiding counter-trend trades.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "12H_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend direction
    close_1w = pd.Series(df_1w['close'])
    # Calculate ER (Efficiency Ratio)
    change = abs(close_1w.diff(10))  # 10-period change
    volatility = abs(close_1w.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama_1w = [np.nan] * len(close_1w)
    if len(close_1w) > 0:
        kama_1w[0] = close_1w.iloc[0]
        for i in range(1, len(close_1w)):
            if np.isnan(sc.iloc[i]) or np.isnan(kama_1w[i-1]):
                kama_1w[i] = kama_1w[i-1]
            else:
                kama_1w[i] = kama_1w[i-1] + sc.iloc[i] * (close_1w.iloc[i] - kama_1w[i-1])
    kama_1w = np.array(kama_1w)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = pd.Series(df_1d['close'])
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # Fill NaN with neutral 50
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: volume > 1.5x 30-period average on 12h chart
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 30)  # Warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly KAMA
        price_above_kama = close[i] > kama_1w_aligned[i]
        price_below_kama = close[i] < kama_1w_aligned[i]
        
        if position == 0:
            # Long entry: price above KAMA + RSI > 50 (bullish momentum) + volume surge
            if (price_above_kama and 
                rsi_1d_aligned[i] > 50 and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + RSI < 50 (bearish momentum) + volume surge
            elif (price_below_kama and 
                  rsi_1d_aligned[i] < 50 and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI < 40 (losing momentum)
            if (not price_above_kama or rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI > 60 (losing momentum)
            if (not price_below_kama or rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals