#!/usr/bin/env python3
"""
#100958 - 1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: Use KAMA direction (trend) on daily timeframe combined with RSI for momentum and Choppiness Index for regime filtering. KAMA adapts to market noise, reducing false signals in chop. RSI identifies overbought/oversold conditions within the trend. Chop filter avoids trend-following in ranging markets. Designed for low trade frequency (7-25/year) to minimize fee drag on 1d timeframe. Works in bull (KAMA up + RSI > 50) and bear (KAMA down + RSI < 50) by following the adaptive trend.
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
    
    # Get weekly data for trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix dimensions: volatility needs to be same length as change
    volatility = np.convolve(np.abs(np.diff(close)), np.ones(10), 'valid')
    # Handle edge cases: pad with zeros for first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # conv reduces by 9
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to daily timeframe (already daily, but ensure alignment)
    # Since we calculated on close directly, no alignment needed for KAMA itself
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first 14 values
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([np.array([np.nan]), tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where((hh - ll) != 0, 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14), 50)
    # Pad first 14 values
    chop = np.concatenate([np.full(14, np.nan), chop[14:]])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above KAMA (uptrend), RSI > 50 (bullish momentum), Chop < 61.8 (trending market)
        if (close[i] > kama[i] and 
            rsi[i] > 50 and 
            chop[i] < 61.8 and
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below KAMA (downtrend), RSI < 50 (bearish momentum), Chop < 61.8 (trending market)
        elif (close[i] < kama[i] and 
              rsi[i] < 50 and 
              chop[i] < 61.8 and
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend change or choppy market
        elif position == 1 and (close[i] < kama[i] or chop[i] > 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > kama[i] or chop[i] > 61.8):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0