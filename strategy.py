#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v3
Hypothesis: Daily KAMA trend direction + RSI(2) extreme + Choppiness Index regime filter. 
KAMA adapts to market noise, reducing false signals in chop. RSI(2) catches short-term overextensions 
within the trend. Choppiness Index > 61.8 prevents trading in strong trends where mean reversion fails. 
Designed for low frequency (15-25 trades/year) to minimize fee drag and work in both bull (trend follow) 
and bear (mean revert in range) markets via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1w for regime context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1d KAMA trend (ER=10, FAST=2, SLOW=30) ===
    close = prices['close'].values
    direction = np.diff(close, prepend=close[0])
    volatility = np.abs(direction)
    er = np.zeros_like(close)
    for i in range(10, n):  # min_periods=10 for ER
        if volatility[i-9:i+1].sum() > 0:
            er[i] = abs(close[i] - close[i-9]) / volatility[i-9:i+1].sum()
        else:
            er[i] = 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smooth constant
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # already LTF
    
    # === 1d RSI(2) for mean reversion extremes ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:2] = np.nan  # not enough data
    
    # === 1w Choppiness Index (14-period) for regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = np.zeros(len(high_1w))
    for i in range(len(high_1w)):
        tr = max(high_1w[i] - low_1w[i], 
                 abs(high_1w[i] - close_1w[i-1]) if i > 0 else 0,
                 abs(low_1w[i] - close_1w[i-1]) if i > 0 else 0)
        atr_1w[i] = tr if i == 0 else (atr_1w[i-1] * 13 + tr) / 14
    atr_sum_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    highest_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum_1w / (highest_1w - lowest_1w)) / np.log10(14)
    chop[0:13] = np.nan
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: Only trade when chop > 61.8 (ranging market)
        if chop_val <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend) + RSI(2) oversold
            if price_close < kama_val and rsi_val < 15:
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (rally in downtrend) + RSI(2) overbought
            elif price_close > kama_val and rsi_val > 85:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price reverts to KAMA or RSI returns to neutral
            if position == 1:
                if price_close >= kama_val or rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close <= kama_val or rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v3"
timeframe = "1d"
leverage = 1.0