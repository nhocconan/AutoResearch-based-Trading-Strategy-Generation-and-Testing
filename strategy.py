#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_ChopFilter
Hypothesis: 12h KAMA trend direction combined with RSI extremes and Choppiness Index regime filter.
Long when KAMA up, RSI<30 (oversold) and CHOP>61.8 (ranging market). Short when KAMA down, RSI>70 (overbought) and CHOP>61.8.
Uses weekly trend filter to avoid major counter-trend moves. Designed for 12-37 trades/year on 12h timeframe.
Works in both bull and bear markets by fading extremes in ranging regimes while respecting weekly trend.
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
    
    # Get 1d data for weekly trend filter (using 1d as proxy for weekly to reduce complexity)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend direction
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation over 10 periods
    er = np.zeros(n)
    for i in range(10, n):
        direction = np.abs(close[i] - close[i-10])
        volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))

        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - momentum oscillator
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) - regime detection
    # CHOP = 100 * log10(sum(ATR) / (log10(highest high - lowest low) * sqrt(n))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros(n)
    for i in range(14, n):
        if hh[i] > ll[i] and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / np.log10(hh[i] - ll[i]) / np.sqrt(14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Weekly trend filter using 1d EMA50 (proxy for weekly)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    weekly_uptrend = close > ema_50_1d_aligned
    weekly_downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA, 14 for RSI/CHOP, 50 for EMA)
    start_idx = max(30, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: KAMA up, RSI oversold, choppy market, weekly uptrend
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop[i] > 61.8 and 
                weekly_uptrend[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, choppy market, weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8 and 
                  weekly_downtrend[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA down OR RSI>50 (mean reversion) OR chop<38.2 (trending) OR weekly trend change
            if (close[i] < kama[i] or rsi[i] > 50 or chop[i] < 38.2 or not weekly_uptrend[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA up OR RSI<50 (mean reversion) OR chop<38.2 (trending) OR weekly trend change
            if (close[i] > kama[i] or rsi[i] < 50 or chop[i] < 38.2 or not weekly_downtrend[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Direction_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0