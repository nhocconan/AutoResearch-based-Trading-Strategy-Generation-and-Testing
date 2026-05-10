#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Trend
# Hypothesis: Daily KAMA trend direction combined with RSI overbought/oversold and Choppiness Index regime filter.
# KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI identifies extreme momentum.
# Choppiness Index > 61.8 indicates ranging (mean reversion), < 38.2 indicates trending (trend follow).
# In trending regimes (CHOP < 38.2): follow KAMA direction.
# In ranging regimes (CHOP > 61.8): fade RSI extremes (sell overbought, buy oversold).
# Volume confirmation (1.5x 20-day average) filters low-conviction moves.
# Designed for 1d timeframe targeting 15-25 trades/year per symbol.
# Works in bull/bear by adapting to market regime via Choppiness Index.

name = "1d_KAMA_RSI_Chop_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period ER, 2/30 SC
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper volatility calculation for ER
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-9):i+1])))
    
    er = np.zeros(n)
    er[:9] = 0  # not enough data
    for i in range(9, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max/min close over 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(13, n):
        if max_close[i] - min_close[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Volume average (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if is_trending:
                # In trending regime: follow KAMA direction
                if close[i] > kama[i] and volume_surge:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and volume_surge:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # In ranging regime: fade RSI extremes
                if rsi[i] < 30 and volume_surge:  # oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and volume_surge:  # overbought
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Long exit conditions
                if is_trending and close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                elif is_ranging and rsi[i] > 50:  # exit mean reversion at midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit conditions
                if is_trending and close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                elif is_ranging and rsi[i] < 50:  # exit mean reversion at midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals