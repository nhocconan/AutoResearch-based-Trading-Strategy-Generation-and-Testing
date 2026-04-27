#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction + RSI filter + chop regime filter
# Uses 12h KAMA for trend direction (long when price > KAMA, short when price < KAMA)
# RSI(14) > 50 for long, < 50 for short to ensure momentum alignment
# Choppiness Index(14) > 61.8 for ranging market (mean reversion at extremes)
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability trends
# Focus on BTC/ETH as primary assets with proven KAMA edge from DB

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h KAMA
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=1)  # 10-period volatility
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_12h = kama
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 12h RSI(14)
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 12h Choppiness Index(14)
    atr_12h = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        tr = max(high_12h[i] - low_12h[i],
                 abs(high_12h[i] - close_12h[i-1]),
                 abs(low_12h[i] - close_12h[i-1]))
        atr_12h[i] = tr
    # Smooth ATR
    atr_ma = pd.Series(atr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Chop calculation
    sum_atr = pd.Series(atr_ma).rolling(window=14, min_periods=14).sum().values
    max_range = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values - \
                pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop = np.where(max_range > 0, 100 * np.log10(sum_atr / max_range) / np.log10(14), 50)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = 30  # enough for KAMA, RSI, Chop
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h_aligned[i]) or
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine trend from 12h KAMA
        uptrend = price > kama_12h_aligned[i]
        downtrend = price < kama_12h_aligned[i]
        
        # RSI filter: >50 for long, <50 for short
        rsi_long = rsi_12h_aligned[i] > 50
        rsi_short = rsi_12h_aligned[i] < 50
        
        # Chop filter: >61.8 for ranging market
        chop_high = chop_12h_aligned[i] > 61.8
        
        if position == 0:
            # Long: uptrend + RSI>50 + chop>61.8 (fading in range)
            if uptrend and rsi_long and chop_high:
                signals[i] = size
                position = 1
            # Short: downtrend + RSI<50 + chop>61.8 (fading in range)
            elif downtrend and rsi_short and chop_high:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below KAMA or RSI<40
            if price < kama_12h_aligned[i] or rsi_12h_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price above KAMA or RSI>60
            if price > kama_12h_aligned[i] or rsi_12h_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0