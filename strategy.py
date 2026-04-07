#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1D KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, reducing false signals in chop.
# RSI provides momentum confirmation. Chop filter avoids trending markets where mean reversion fails.
# Works in bull/bear by adapting to volatility regime. Target: 10-25 trades/year.
name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for chop filter (regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA (Adaptive Moving Average) on daily
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will fix below
    # Recalculate volatility properly
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.zeros(n)
    er[10:] = change[10:] / (volatility[10:] + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on daily
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Chopiness Index on weekly (trend vs range detection)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    atr_1w = np.zeros(len(weekly_high))
    tr_1w = np.zeros(len(weekly_high))
    for i in range(1, len(weekly_high)):
        tr_1w[i] = max(
            weekly_high[i] - weekly_low[i],
            abs(weekly_high[i] - weekly_close[i-1]),
            abs(weekly_low[i] - weekly_close[i-1])
        )
    # Simplified ATR calculation
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    # Sum of true ranges over 14 periods
    sum_tr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    # Max/min range over 14 periods
    max_high_1w = pd.Series(weekly_high).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(weekly_low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_1w / (max_high_1w - min_low_1w + 1e-10)) / np.log10(14)
    chop_1d = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1d[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is ranging (chop > 61.8)
        if chop_1d[i] > 61.8:
            # Mean reversion in ranging markets
            if close[i] < kama[i] and rsi[i] < 40:
                signals[i] = 0.25  # Long
            elif close[i] > kama[i] and rsi[i] > 60:
                signals[i] = -0.25  # Short
    
    return signals