#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Reversal_v2"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # KAMA on daily close
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(tr, axis=1) / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) != 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > KAMA, RSI < 30, Chop > 61.8 (range), weekly uptrend
            if close[i] > kama[i] and rsi[i] < 30 and chop[i] > 61.8 and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA, RSI > 70, Chop > 61.8 (range), weekly downtrend
            elif close[i] < kama[i] and rsi[i] > 70 and chop[i] > 61.8 and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < KAMA or RSI > 70 or chop < 38.2 (trend) or weekly trend down
            if close[i] < kama[i] or rsi[i] > 70 or chop[i] < 38.2 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > KAMA or RSI < 30 or chop < 38.2 (trend) or weekly trend up
            if close[i] > kama[i] or rsi[i] < 30 or chop[i] < 38.2 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA + RSI + Chop filter on 1d with 1w EMA34 trend filter.
# Long when price > KAMA (trend), RSI < 30 (oversold), Chop > 61.8 (range), and weekly uptrend.
# Short when price < KAMA (trend), RSI > 70 (overbought), Chop > 61.8 (range), and weekly downtrend.
# Uses Chop filter to avoid whipsaws in strong trends, focusing on mean reversion in ranges.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed for 1d timeframe to target 30-100 total trades over 4 years, avoiding overtrading.
# Works in both bull and bear markets by fading extremes in ranging conditions.