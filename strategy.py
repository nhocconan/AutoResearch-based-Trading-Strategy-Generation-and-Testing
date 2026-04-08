#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter v1
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) captures the adaptive trend direction,
RSI(14) filters for momentum extremes, and Choppiness Index (CHOP) identifies ranging vs trending regimes.
In trending regimes (CHOP < 38.2), we follow KAMA direction; in ranging regimes (CHOP > 61.8), we fade RSI extremes.
This adapts to both bull and bear markets by switching between trend-following and mean-reversion based on market state.
Timeframe: 1d targets 7-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1d Indicators ===
    # KAMA components
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan
    volatility = np.abs(np.diff(close, prepend=np.nan))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = change / volatility_sum
    er = np.where(volatility_sum == 0, 0, er)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP)
    atr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)
    
    # === Weekly Trend Filter (1w) ===
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = df_1w['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend change or reversal signal
            if close[i] < kama[i] or (chop[i] > 61.8 and rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend change or reversal signal
            if close[i] > kama[i] or (chop[i] > 61.8 and rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine regime
            if chop[i] < 38.2:  # Trending regime - follow KAMA
                if close[i] > kama[i] and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < kama[i] and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif chop[i] > 61.8:  # Ranging regime - fade RSI extremes
                if rsi[i] < 30 and close[i] > kama[i]:
                    position = 1
                    signals[i] = 0.20
                elif rsi[i] > 70 and close[i] < kama[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals