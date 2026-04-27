#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) to determine trend direction on 12h, combined with RSI for momentum confirmation and Choppiness Index to filter range-bound markets. Designed for low trade frequency (<30/year) to minimize fee drag, working in trending markets by requiring trend alignment and momentum, while avoiding whipsaws in ranging conditions.
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
    
    # Calculate KAMA on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = abs(df_12h['close'].diff(er_length))
    volatility = df_12h['close'].diff().abs().rolling(window=er_length).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(df_12h))
    kama[0] = df_12h['close'].iloc[0]
    for i in range(1, len(df_12h)):
        kama[i] = kama[i-1] + sc.iloc[i] * (df_12h['close'].iloc[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (wait for previous 12h bar's close)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate RSI on 12h
    delta = df_12h['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi.values)
    
    # Calculate Choppiness Index on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = df_1d['high'].rolling(window=14, min_periods=14).max()
    ll = df_1d['low'].rolling(window=14, min_periods=14).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        
        # Chop filter: only trade when market is trending (CHOP < 38.2)
        if chop_val > 38.2:
            # In ranging market, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA and RSI > 50 (bullish momentum)
            if close_val > kama_val and rsi_val > 50:
                signals[i] = size
                position = 1
            # Short: price below KAMA and RSI < 50 (bearish momentum)
            elif close_val < kama_val and rsi_val < 50:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or RSI < 40 (loss of momentum)
            if close_val < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above KAMA or RSI > 60 (loss of momentum)
            if close_val > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0