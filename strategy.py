#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend direction with RSI extremes and weekly choppiness regime filter.
Long when KAMA is rising, RSI < 30 (oversold), and weekly chop < 61.8 (trending regime).
Short when KAMA is falling, RSI > 70 (overbought), and weekly chop < 61.8 (trending regime).
Exit when RSI returns to neutral (40-60) or chop > 61.8 (range regime).
Uses 1d for execution and RSI, 1w for KAMA trend and chop regime.
Designed to capture mean-reversion moves within trending regimes across bull and bear markets.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # neutral for first value
    
    # Get 1w data for KAMA and chop regime
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w KAMA(10,2,30)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1w, n=10, prepend=close_1w[:10]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0) if len(close_1w) > 1 else 0
    # Simplified volatility calculation for 10-period
    volatility_10 = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):
        volatility_10[i] = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
    er = np.where(volatility_10 > 0, change / volatility_10, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Calculate 1w Choppiness Index(14)
    # True Range
    tr1 = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr2 = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr3 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3[0] = 0
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/min close over 14 periods
    max_close = pd.Series(close_1w).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1w).rolling(window=14, min_periods=14).min().values
    # Choppy Index
    chop = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        if atr_sum[i] > 0 and max_close[i] != min_close[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    chop[:14] = 50  # neutral for insufficient data
    
    # Align all indicators to primary timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(kama_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction (using 1-period change)
        kama_rising = kama_aligned[i] > kama_aligned[i-1] if i > 0 else False
        kama_falling = kama_aligned[i] < kama_aligned[i-1] if i > 0 else False
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        rsi_neutral = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        # Chop regime: trending when chop < 61.8
        chop_trending = chop_aligned[i] < 61.8
        chop_range = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, trending regime
            if kama_rising and rsi_oversold and chop_trending:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, trending regime
            elif kama_falling and rsi_overbought and chop_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral OR regime changes to range
            if rsi_neutral or chop_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral OR regime changes to range
            if rsi_neutral or chop_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0