#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily KAMA direction + RSI + chop regime filter
# KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets
# RSI(14) > 50 for long, < 50 for short filters counter-trend noise
# Choppiness index (CHOP) > 61.8 defines ranging market (fade extremes), < 38.2 defines trending (follow momentum)
# Combines adaptive trend, momentum filter, and regime detection for robustness across bull/bear markets
# Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "4h_KAMA_RSI_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily KAMA direction ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Kaufman's Adaptive Moving Average (KAMA)
    close_1d = df_1d['close'].values
    direction = np.abs(close_1d[1:] - close_1d[:-1])
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 1
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i-1] * (close_1d[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.where(kama[1:] > kama[:-1], 1, -1)
    kama_dir = np.concatenate([[1], kama_dir])  # first bar assumes up
    
    # Align daily KAMA direction to 4h timeframe
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on 4h data (14-period)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_14[0:13] = np.nan  # first 13 values invalid
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (highest_high - lowest_low) != 0,
        100 * np.log10(atr_14.sum() / (highest_high - lowest_low)) / np.log10(14),
        50
    )
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, and not in strong chop (CHOP <= 61.8)
            if kama_dir_aligned[i] == 1 and rsi[i] > 50 and chop[i] <= 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, and not in strong chop (CHOP <= 61.8)
            elif kama_dir_aligned[i] == -1 and rsi[i] < 50 and chop[i] <= 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down OR RSI < 40
            if kama_dir_aligned[i] == -1 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up OR RSI > 60
            if kama_dir_aligned[i] == 1 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals