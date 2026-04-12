#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_filter_v1
Hypothesis: Daily strategy using Kaufman's Adaptive Moving Average (KAMA) for trend direction,
RSI for overbought/oversold conditions, and Choppiness Index for regime filtering.
KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI identifies extreme
momentum, while Choppiness Index filters out trades in overly choppy (range-bound) or
strongly trending regimes, focusing on transitions. Works in bull/bear by adapting trend
detection and avoiding false signals in non-trending conditions.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
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
    
    # Calculate KAMA on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Set first TR to high-low (no previous close)
    tr[0] = high[0] - low[0]
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max and min close over 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    # Choppy Index
    chop = np.where(
        (atr_sum > 0) & (max_close != min_close),
        100 * np.log10(atr_sum / (max_close - min_close)) / np.log10(14),
        50  # Neutral if undefined
    )
    
    # Get weekly data for trend filter (optional, but can add robustness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # Fallback to daily EMA50 if weekly data insufficient
        ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
        trend_filter = close > ema50  # Simple uptrend filter
    else:
        close_1w = df_1w['close'].values
        ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
        trend_filter = close > ema20_1w_aligned  # Use weekly EMA20 for trend
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after indicators are ready
        # Skip if any key data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(trend_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above KAMA (uptrend), RSI oversold (<30), not too choppy (<61.8)
        if (close[i] > kama[i] and rsi[i] < 30 and chop[i] < 61.8 and trend_filter[i]):
            signals[i] = 0.25
        # Short conditions: price below KAMA (downtrend), RSI overbought (>70), not too choppy (<61.8)
        elif (close[i] < kama[i] and rsi[i] > 70 and chop[i] < 61.8 and not trend_filter[i]):
            signals[i] = -0.25
        # Exit conditions: RSI returns to neutral (40-60) or market becomes too choppy
        elif (40 <= rsi[i] <= 60) or chop[i] > 61.8:
            signals[i] = 0.0
        else:
            # Hold previous signal to avoid unnecessary changes
            signals[i] = signals[i-1]
    
    return signals

name = "1d_1w_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0