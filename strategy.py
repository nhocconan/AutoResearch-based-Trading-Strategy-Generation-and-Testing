#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend direction with RSI momentum filter and weekly choppiness regime filter.
Long when KAMA is rising (bullish trend) AND RSI < 30 (oversold bounce) AND weekly chop < 38.2 (trending regime).
Short when KAMA is falling (bearish trend) AND RSI > 70 (overbought bounce) AND weekly chop < 38.2 (trending regime).
Exit when RSI crosses 50 (mean reversion) or chop > 61.8 (range regime).
Uses 1w for choppiness regime filter and 1d for execution.
Designed to catch mean-reversion bounces within strong trends across bull and bear markets.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA (10-period)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly: sum of absolute changes over 10 periods
    volatility = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for choppiness regime
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Choppiness Index (14-period)
    # True Range
    tr1 = np.maximum(high_1w - low_1w, 
                     np.absolute(high_1w - np.roll(close_1w, 1)),
                     np.absolute(low_1w - np.roll(close_1w, 1)))
    tr1[0] = high_1w[0] - low_1w[0]
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14)
    
    # Align all indicators to primary timeframe (1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising or falling
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        rsi_mean = abs(rsi_aligned[i] - 50) < 1  # near 50 for exit
        
        # Chop regime: trending (< 38.2) or ranging (> 61.8)
        chop_trending = chop_aligned[i] < 38.2
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, trending regime
            if (kama_rising and rsi_oversold and chop_trending):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, trending regime
            elif (kama_falling and rsi_overbought and chop_trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI mean reversion or chop becomes ranging
            if (rsi_mean or chop_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI mean reversion or chop becomes ranging
            if (rsi_mean or chop_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0