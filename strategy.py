#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily KAMA (Kaufman Adaptive Moving Average) for trend direction ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = abs(close_1d_series - close_1d_series.shift(10))
    volatility = abs(close_1d_series - close_1d_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === Daily RSI(14) for overbought/oversold ===
    delta = close_1d_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 12h: Choppiness Index (CHOP) regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop calculation
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((hh - ll) > 0, chop, 50.0)  # Default to neutral if range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for daily indicators
        # Get values
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop[i]
        
        # Skip if any value is NaN
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up (price > KAMA), RSI < 40 (oversold), CHOP > 61.8 (ranging market)
            if close[i] > kama_val and rsi_val < 40 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (price < KAMA), RSI > 60 (overbought), CHOP > 61.8 (ranging market)
            elif close[i] < kama_val and rsi_val > 60 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below KAMA or RSI > 70 (overbought)
            if close[i] < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above KAMA or RSI < 30 (oversold)
            if close[i] > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals