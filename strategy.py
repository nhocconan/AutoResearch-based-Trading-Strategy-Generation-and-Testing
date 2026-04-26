#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v2
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets. RSI filters overextended conditions, while Choppiness Index identifies ranging vs trending regimes. In trending regimes (CHOP < 38.2), follow KAMA direction. In ranging regimes (CHOP > 61.8), fade extreme RSI readings. Volume confirmation ensures institutional participation. Designed for 4h timeframe with discrete position sizing to minimize fee drag while capturing sustained moves.
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF regime filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h close
    # Efficiency Ratio = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, n=1))
    volatility = np.sum(change.reshape(-1, 10), axis=1)  # 10-period volatility
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align lengths
    er = np.abs(np.diff(close, n=10)) / volatility
    er = np.concatenate([np.full(10, np.nan), er])  # align lengths
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # align
    
    # Calculate Choppiness Index on 1d data
    if len(df_1d) < 20:
        return np.zeros(n)
    atr_1d = []
    for i in range(1, len(df_1d)):
        tr = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                 abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                 abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    atr_1d = np.concatenate([[np.nan], atr_1d])  # align
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_h = df_1d['high'].rolling(window=14, min_periods=14).max().values
    min_l = df_1d['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_h - min_l + 1e-10)) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > volume_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filters
        trending = chop_aligned[i] < 38.2
        ranging = chop_aligned[i] > 61.8
        
        # Volume confirmation
        vol_confirmed = volume_ok[i]
        
        # Long logic
        long_signal = False
        if trending and vol_confirmed:
            # In trending regime, follow KAMA direction
            if close[i] > kama[i]:
                long_signal = True
        elif ranging and vol_confirmed:
            # In ranging regime, fade extreme RSI (mean reversion)
            if rsi[i] < 30:  # oversold
                long_signal = True
        
        # Short logic
        short_signal = False
        if trending and vol_confirmed:
            # In trending regime, follow KAMA direction
            if close[i] < kama[i]:
                short_signal = True
        elif ranging and vol_confirmed:
            # In ranging regime, fade extreme RSI (mean reversion)
            if rsi[i] > 70:  # overbought
                short_signal = True
        
        # Exit logic: opposite signal or regime change to extreme chop
        exit_long = position == 1 and (short_signal or chop_aligned[i] > 61.8)
        exit_short = position == -1 and (long_signal or chop_aligned[i] > 61.8)
        
        if long_signal and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_signal and position != -1:
            signals[i] = -0.25
            position = -1
        elif exit_long or exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0