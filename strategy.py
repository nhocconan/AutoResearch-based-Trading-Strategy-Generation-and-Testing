#!/usr/bin/env python3
# 12H_1D_Retracement_Signal
# Hypothesis: In strong daily trends, price retraces to EMA21 before continuing.
# Long when price crosses above EMA21 in a daily uptrend (close > EMA50).
# Short when price crosses below EMA21 in a daily downtrend (close < EMA50).
# Uses 1d EMA50 for trend filter and 1d EMA21 for retracement entry.
# Works in bull/bear by following daily trend direction. Target: 15-25 trades/year per symbol.

name = "12H_1D_Retracement_Signal"
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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily EMA21 for retracement entry
    ema21_1d = close_1d_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align to 12h
    ema21_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + price crosses above daily EMA21
            if bullish and close[i] > ema21_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price crosses below daily EMA21
            elif bearish and close[i] < ema21_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price crosses below daily EMA50
            ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
            if bearish or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price crosses above daily EMA50
            ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
            if bullish or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals