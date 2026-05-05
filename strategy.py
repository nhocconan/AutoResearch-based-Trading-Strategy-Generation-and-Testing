#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) mean reversion + chop regime filter
# KAMA adapts to market noise, effective in both trending and ranging markets.
# Long when KAMA direction up AND RSI < 40 (oversold) AND chop > 61.8 (ranging market)
# Short when KAMA direction down AND RSI > 60 (overbought) AND chop > 61.8 (ranging market)
# Exit when RSI reverts to 50 or chop < 38.2 (trending market)
# Uses daily timeframe to minimize fees, chop filter to avoid false signals in strong trends.
# Timeframe: 1d, HTF: 1w for trend confirmation. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_KAMA_RSI_ChopRegime_1wEMA50"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on 1d
    if len(close) >= 10:
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=10))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 10 else np.zeros_like(change)
        # Handle first 10 values
        er = np.full_like(change, 0.0, dtype=float)
        valid_idx = volatility != 0
        er[valid_idx] = change[valid_idx] / volatility[valid_idx][-len(change):] if len(volatility) >= len(change) else change[valid_idx] / np.maximum(volatility[-len(change):], 1e-10)
        er = np.concatenate([np.full(10, 0.0), er])
        # Smoothing constants
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        sc = er * (fast_sc - slow_sc) + slow_sc
        sc = sc ** 2
        # KAMA calculation
        kama = np.full_like(close, np.nan, dtype=float)
        kama[9] = close[9]  # seed
        for i in range(10, n):
            if not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = close[i]
        kama_dir = np.diff(kama, prepend=kama[0]) > 0
    else:
        kama = close.copy()
        kama_dir = np.zeros(n, dtype=bool)
    
    # Calculate RSI(14) on 1d
    if len(close) >= 15:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.full(n, 50.0)
    
    # Calculate Choppiness Index(14) on 1d
    if len(close) >= 15:
        atr_14 = np.zeros(n)
        tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14 if i >= 1 else tr[i]
        atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
        max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        chop = np.where(atr_sum > 0, 100 * np.log10(max_high - min_low) / np.log10(14) / np.log10(atr_sum), 50)
        chop = np.nan_to_num(chop, nan=50.0)
    else:
        chop = np.full(n, 50.0)
    
    # Get 1w data ONCE before loop for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up AND RSI < 40 AND chop > 61.8 AND above 1w EMA50
            if (kama_dir[i] and 
                rsi[i] < 40 and 
                chop[i] > 61.8 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down AND RSI > 60 AND chop > 61.8 AND below 1w EMA50
            elif (not kama_dir[i] and 
                  rsi[i] > 60 and 
                  chop[i] > 61.8 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI reverts to 50 OR chop < 38.2 (trending market)
            if rsi[i] >= 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI reverts to 50 OR chop < 38.2 (trending market)
            if rsi[i] <= 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals