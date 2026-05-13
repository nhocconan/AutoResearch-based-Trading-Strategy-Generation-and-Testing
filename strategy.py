#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop
Hypothesis: KAMA identifies adaptive trend direction, RSI measures momentum strength, and Choppiness Index filters range vs trend. 
Long when KAMA trending up, RSI > 50, and CHOP > 61.8 (range). Short when KAMA trending down, RSI < 50, and CHOP > 61.8.
Mean reversion in ranging markets (CHOP high) with trend filter (KAMA) and momentum confirmation (RSI).
Works in sideways markets (2025-2026) and avoids strong trends via CHOP filter.
Target: 10-25 trades/year per symbol.
"""

name = "1d_KAMA_RSI_Chop"
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
    
    # Kaufman Adaptive Moving Average (KAMA)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (CHOP) - 14 period
    atr = pd.Series(np.sqrt((high - low)**2)).rolling(window=14, min_periods=14).mean()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # Weekly trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: KAMA up, RSI > 50, CHOP > 61.8 (range), 1w uptrend
            if close[i] > kama_val and rsi_val > 50 and chop_val > 61.8 and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI < 50, CHOP > 61.8 (range), 1w downtrend
            elif close[i] < kama_val and rsi_val < 50 and chop_val > 61.8 and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down or RSI <= 50
            if close[i] < kama_val or rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up or RSI >= 50
            if close[i] > kama_val or rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals