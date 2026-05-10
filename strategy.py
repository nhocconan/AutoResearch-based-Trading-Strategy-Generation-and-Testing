#!/usr/bin/env python3
# 12H_1D_RSI_Retracement_Trend_Filter
# Hypothesis: In strong daily trends, price pulls back to RSI levels before continuing.
# Long when RSI(14) crosses above 30 in daily uptrend (close > EMA50).
# Short when RSI(14) crosses below 70 in daily downtrend (close < EMA50).
# Uses 1d EMA50 for trend filter and 1d RSI(14) for retracement entry.
# Works in bull/bear by following daily trend direction. Target: 15-25 trades/year per symbol.

name = "12H_1D_RSI_Retracement_Trend_Filter"
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
    
    # Daily RSI(14) for retracement entry
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + RSI crosses above 30
            if bullish and rsi_aligned[i] > 30 and rsi_aligned[i-1] <= 30:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + RSI crosses below 70
            elif bearish and rsi_aligned[i] < 70 and rsi_aligned[i-1] >= 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or RSI crosses below 50
            if bearish or (rsi_aligned[i] < 50 and rsi_aligned[i-1] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or RSI crosses above 50
            if bullish or (rsi_aligned[i] > 50 and rsi_aligned[i-1] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals