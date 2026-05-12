#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: Use KAMA trend direction on daily timeframe for primary signal, filtered by RSI extremes and Choppiness index to avoid whipsaws in ranging markets.
# Long when KAMA rising AND RSI > 50 AND Choppiness > 61.8 (ranging market) OR Choppiness < 38.2 (trending market).
# Short when KAMA falling AND RSI < 50 AND same Choppiness condition.
# Exit when KAMA direction reverses or RSI reaches opposite extreme.
# Designed for low frequency (10-25 trades/year) by using daily timeframe with multiple confirmation filters.

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # === 1w data for Choppiness index (regime filter) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index on weekly data
    # CHOP = 100 * log10(SUM(ATR1) / (n * (MAX(HIGH) - MIN(LOW)))) / log10(n)
    atr_1w = np.maximum(high_1w - low_1w, 
                        np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                                   np.abs(low_1w - np.roll(close_1w, 1))))
    atr_1w[0] = high_1w[0] - low_1w[0]  # first value
    
    tr_sum_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    chop_1w = 100 * np.log10(tr_sum_14 / (14 * (max_high_14 - min_low_14))) / np.log10(14)
    chop_1w = np.where((max_high_14 - min_low_14) == 0, 50, chop_1w)  # avoid division by zero
    
    # Align weekly Choppiness to daily timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === 1d data for KAMA and RSI ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio = |net change| / sum(|changes|)
    # Smoothing Constant = [ER * (fastest SC - slowest SC) + slowest SC]^2
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    direction = np.abs(np.subtract(close_1d, np.roll(close_1d, 10)))  # 10-period net change
    er = np.where(change != 0, direction / np.convolve(change, np.ones(10), 'same'), 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2  # fast=2/(2+1), slow=2/(30+1)
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily indicators to daily timeframe (no alignment needed as already daily)
    kama_aligned = kama  # already daily
    rsi_aligned = rsi    # already daily
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(chop_1w_aligned[i]) or np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI condition: >50 for bullish bias, <50 for bearish bias
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Choppiness regime: 
        # - Chop > 61.8: ranging market (mean reversion favorable)
        # - Chop < 38.2: trending market (trend following favorable)
        chop_value = chop_1w_aligned[i]
        chop_ranging = chop_value > 61.8
        chop_trending = chop_value < 38.2
        chop_ok = chop_ranging or chop_trending  # avoid extremely choppy middle (38.2-61.8)
        
        if position == 0:
            # LONG: KAMA rising AND RSI > 50 AND chop regime OK
            if kama_rising and rsi_bullish and chop_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling AND RSI < 50 AND chop regime OK
            elif kama_falling and rsi_bearish and chop_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI < 30 (oversold)
            if not kama_rising or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI > 70 (overbought)
            if not kama_falling or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals