#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction with RSI(14) mean reversion entries and choppiness regime filter.
# Long when KAMA is rising (bullish trend) AND RSI < 30 (oversold) AND choppiness > 61.8 (range market)
# Short when KAMA is falling (bearish trend) AND RSI > 70 (overbought) AND choppiness > 61.8 (range market)
# Exit when RSI crosses 50 (mean reversion completion) OR opposite signal fires
# Uses 1d timeframe with choppiness filter to avoid trending markets where mean reversion fails.
# KAMA adapts to volatility, reducing whipsaws in choppy markets like 2022 and 2025.
# Target: 30-100 trades over 4 years by combining trend filter with precise mean reversion entries.

name = "1d_KAMA_RSI_Chop_MR_v1"
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
    
    # Calculate KAMA (Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Handle array dimensions correctly
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(0, np.nan), volatility]) if len(volatility) > 0 else np.array([])
    # Recalculate volatility properly: sum of |close[i] - close[i-1]| over 10 periods
    volatility_10 = pd.Series(close).diff().abs().rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_10 > 0, change_padded / volatility_10, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first ER calculation
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, n=1)
    delta = np.concatenate([[np.nan], delta])  # align with close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where((hh_14 - ll_14) != 0, 
                    100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14), 
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            i == 0 or np.isnan(kama[i-1])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (bullish trend) AND RSI < 30 (oversold) AND chop > 61.8 (range)
            if kama[i] > kama[i-1] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (bearish trend) AND RSI > 70 (overbought) AND chop > 61.8 (range)
            elif kama[i] < kama[i-1] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals