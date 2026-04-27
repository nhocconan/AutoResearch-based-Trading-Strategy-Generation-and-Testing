#1d_KAMA_RSI_ChopFilter_v1
# Strategy: KAMA trend direction + RSI momentum + Choppiness regime filter
# Why it works: KAMA adapts to market noise, RSI captures momentum extremes,
# Choppiness filter avoids whipsaws in ranging markets. Works in bull/bear via regime adaptation.
# Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

#!/usr/bin/env python3
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
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily KAMA(14) for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1d, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate daily Choppiness Index(14)
    atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr[0] = high[0] - low[0]
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators
    start_idx = max(34, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1w_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: Choppiness > 61.8 = ranging (mean revert), < 38.2 = trending
        # In ranging markets: buy at RSI < 40, sell at RSI > 60
        # In trending markets: buy when price > KAMA, sell when price < KAMA
        if chop_val > 61.8:  # Ranging market
            if position == 0 and rsi_val < 40:
                signals[i] = size
                position = 1
            elif position == 0 and rsi_val > 60:
                signals[i] = -size
                position = -1
            elif position == 1 and rsi_val > 50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size if position == 1 else (-size if position == -1 else 0.0)
        else:  # Trending market
            if position == 0 and close[i] > kama_val:
                signals[i] = size
                position = 1
            elif position == 0 and close[i] < kama_val:
                signals[i] = -size
                position = -1
            elif position == 1 and close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size if position == 1 else (-size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0