#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter_v2
# Hypothesis: KAMA trend direction on daily + RSI extreme + chop regime filter reduces whipsaw.
# Uses 1w trend filter to avoid counter-trend trades. Target: 15-25 trades/year.
# Works in bull/bear by only trading with weekly trend and avoiding range-bound markets.

name = "1d_KAMA_RSI_ChopFilter_v2"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # ER = Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1d = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First true range
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = np.where(hh_ll > 0, 100 * np.log10(sum_atr14 / hh_ll) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend), RSI < 30 (oversold), chop > 61.8 (ranging)
            if close[i] > kama_1d[i] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend), RSI > 70 (overbought), chop > 61.8 (ranging)
            elif close[i] < kama_1d[i] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend change or RSI overbought
            if close[i] < kama_1d[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend change or RSI oversold
            if close[i] > kama_1d[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals