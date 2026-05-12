#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: Use 1d KAMA (Kaufman Adaptive Moving Average) for trend direction,
# combined with RSI overbought/oversold levels and Choppiness Index regime filter.
# KAMA adapts to market noise, reducing false signals in sideways markets.
# RSI provides mean-reversion signals at extremes.
# Choppiness Index filters trades: only take mean-reversion signals in high-chop (ranging) markets,
# and trend-following signals in low-chop (trending) markets.
# Designed to work in both bull and bear markets by adapting to regime.

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # === 1w KAMA for trend filter (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # KAMA parameters
    fast_ema = 2 / (2 + 1)  # EMA(2)
    slow_ema = 2 / (30 + 1) # EMA(30)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0)
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constant
    sc = np.square(er * (fast_ema - slow_ema) + slow_ema)
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # === 1d RSI for mean reversion ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Choppiness Index for regime filter ===
    # Chop(14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_max_min = max_high - min_low
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(range_max_min > 0, 100 * np.log10(sum_atr / range_max_min) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w KAMA
        trend_up = close[i] > kama_1w_aligned[i]
        trend_down = close[i] < kama_1w_aligned[i]
        
        # Mean reversion signals from RSI
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Regime filter: Choppiness Index
        # Chop > 61.8 = ranging (mean revert)
        # Chop < 38.2 = trending (trend follow)
        chop_high = chop[i] > 61.8  # ranging market
        chop_low = chop[i] < 38.2   # trending market
        
        if position == 0:
            # LONG: mean reversion in ranging market OR trend continuation in trending market
            if (rsi_oversold and chop_high) or (trend_up and chop_low):
                signals[i] = 0.25
                position = 1
            # SHORT: mean reversion in ranging market OR trend continuation in trending market
            elif (rsi_overbought and chop_high) or (trend_down and chop_low):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI overbought or trend breakdown
            if rsi_overbought or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or trend breakdown
            if rsi_oversold or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals