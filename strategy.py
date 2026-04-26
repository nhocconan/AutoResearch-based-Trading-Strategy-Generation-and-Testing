#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_V1
Hypothesis: 1d KAMA trend direction + RSI(14) extreme + Choppiness Index regime filter.
Long when KAMA trending up, RSI < 30 (oversold), and market is choppy (CHOP > 61.8).
Short when KAMA trending down, RSI > 70 (overbought), and market is choppy (CHOP > 61.8).
Uses weekly trend filter to avoid counter-trend trades in strong trends.
Designed for low frequency (15-30 trades/year) to minimize fee drag on 1d timeframe.
Works in both bull and bear markets by combining trend-following with mean reversion in choppy regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # KAMA calculation (1d)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[29] = close_1d[29]  # start after 30 periods
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi_padded = np.full_like(close_1d, np.nan, dtype=np.float64)
    rsi_padded[14:] = rsi
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_padded)
    
    # Choppiness Index (14) on 1d
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    # Max/Min close over 14 periods
    max_close = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(atr14) / (max_close - min_close)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    denominator = max_close - min_close
    chop = np.where(denominator > 0, 100 * np.log10(sum_atr / denominator) / np.log10(14), 100)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Weekly trend filter: price > weekly EMA20 for uptrend, < for downtrend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    uptrend_1w = close > ema_20_1w_aligned
    downtrend_1w = close < ema_20_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA, 14 for RSI/Chop, 20 for weekly EMA)
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: KAMA up, RSI oversold, choppy market, weekly uptrend
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and 
                chop_aligned[i] > 61.8 and 
                uptrend_1w[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, choppy market, weekly downtrend
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  chop_aligned[i] > 61.8 and 
                  downtrend_1w[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR RSI > 50 (mean reversion complete) OR weekly trend changes
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] > 50 or 
                not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR RSI < 50 (mean reversion complete) OR weekly trend changes
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] < 50 or 
                not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_V1"
timeframe = "1d"
leverage = 1.0