#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d KAMA with RSI and chop filter.
# Long when KAMA is rising (bullish trend), RSI < 30 (oversold), and chop > 61.8 (ranging market).
# Short when KAMA is falling (bearish trend), RSI > 70 (overbought), and chop > 61.8 (ranging market).
# Exit when RSI returns to 50 or chop < 38.2 (trending market).
# Uses Kaufman Adaptive Moving Average for trend direction, RSI for mean reversion,
# and Choppiness Index for regime detection. Designed to work in ranging markets
# by fading extremes while avoiding strong trends. Target: 20-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10, 2, 30)
    # Efficiency Ratio
    change = np.abs(close_1d[9:] - close_1d[:-9])
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    er = np.zeros_like(close_1d)
    er[9:] = change / volatility
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi])
    
    # Calculate Choppiness Index (14)
    atr = []
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr.append(tr)
    atr = np.array(atr)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    chop = np.concatenate([[np.nan] * 14, chop])
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 14)  # Need KAMA, RSI, and Chop periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
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
        rsi_neutral = abs(rsi_aligned[i] - 50) < 5
        
        # Chop conditions
        chop_ranging = chop_aligned[i] > 61.8
        chop_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Look for mean reversion in ranging market
            # Long: KAMA rising, RSI oversold, chop ranging
            if (kama_rising and 
                rsi_oversold and 
                chop_ranging):
                position = 1
                signals[i] = position_size
            # Short: KAMA falling, RSI overbought, chop ranging
            elif (kama_falling and 
                  rsi_overbought and 
                  chop_ranging):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or chop becomes trending
            if (rsi_neutral or 
                chop_trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral or chop becomes trending
            if (rsi_neutral or 
                chop_trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_KAMA_RSI_Chop_Fade_v1"
timeframe = "4h"
leverage = 1.0