#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w KAMA(10) for trend filter
    # KAMA calculation: efficiency ratio (ER), smoothing constants
    change = np.abs(np.diff(df_1w['close'], prepend=df_1w['close'][0]))
    volatility = np.abs(np.diff(df_1w['close'])).rolling(window=10, min_periods=10).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(df_1w['close'])
    kama[0] = df_1w['close'][0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1w['close'][i] - kama[i-1])
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(highest_high - lowest_low) / (np.log10(atr * 14) + np.log10(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI < 30 (oversold), chop > 61.8 (range)
            if close[i] > kama_1w_aligned[i] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI > 70 (overbought), chop > 61.8 (range)
            elif close[i] < kama_1w_aligned[i] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 50 or chop < 38.2 (trending)
            if rsi[i] > 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 or chop < 38.2 (trending)
            if rsi[i] < 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d KAMA trend filter with RSI mean reversion in choppy markets
# - KAMA(10) on weekly timeframe identifies dominant trend direction
# - RSI(14) extremes (<30/>70) provide mean-reversion entries within the trend
# - Choppiness Index > 61.8 ensures ranging conditions suitable for mean reversion
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Position size 0.25 targets ~15-25 trades/year, minimizing fee drag
# - Exit when RSI returns to neutral (50) or market starts trending (chop < 38.2)