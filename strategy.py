#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction + RSI(14) + chop filter
# - KAMA identifies adaptive trend direction
# - RSI(14) > 50 for long, < 50 for short to avoid counter-trend
# - Chop filter: avoid extremes (Chop < 38.2 or > 61.8) to trade only in strong trends/ranges
# - Uses 1d for KAMA/RSI/chop calculation for stability
# - Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ER (Efficiency Ratio) for KAMA
    change = np.abs(np.diff(close_1d, 10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    vol = np.concatenate([np.full(10, np.nan), vol])
    er = np.where(vol != 0, change / vol, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop Index (14)
    atr1 = np.abs(high_1d - low_1d)
    atr2 = np.abs(high_1d - np.roll(close_1d, 1))
    atr3 = np.abs(low_1d - np.roll(close_1d, 1))
    atr2[0] = atr1[0]
    atr3[0] = atr1[0]
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.where(max_high - min_low != 0, max_high - min_low, 1)) / np.log10(14)
    
    # Align indicators to 12h timeframe
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h price data
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]) or np.isnan(chop_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_12h[i]
        rsi_val = rsi_12h[i]
        chop_val = chop_12h[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, Chop not extreme
            if price > kama_val and rsi_val > 50 and 38.2 <= chop_val <= 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, Chop not extreme
            elif price < kama_val and rsi_val < 50 and 38.2 <= chop_val <= 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA OR RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA OR RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0