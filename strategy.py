#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop regime
- KAMA direction: long when price > KAMA, short when price < KAMA
- RSI filter: long only when RSI(14) > 50, short only when RSI(14) < 50
- Chop regime: trade only when Chop(14) > 61.8 (range-bound) to avoid whipsaw in trends
- Position size: 0.25
- Exit when KAMA direction flips or Chop < 38.2 (trending regime)
- Uses 1w trend filter: only trade in direction of weekly EMA(20)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (20, 2, 30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    vol = change.rolling(window=30, min_periods=30).sum()
    er = abs(close_s - close_s.shift(9)) / vol.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_s.iloc[0]]
    for i in range(1, len(close_s)):
        kama.append(kama[-1] + sc.iloc[i] * (close_s.iloc[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Chop(14)
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - close_s.shift()), abs(low - close_s.shift()))))
    tr_sum = atr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.values
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i]):
            continue
        
        # Regime filter: only trade in ranging market (Chop > 61.8)
        if chop[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly EMA
        trend_up = close[i] > ema_1w_aligned[i]
        
        # KAMA direction + RSI filter
        if close[i] > kama[i] and rsi[i] > 50 and trend_up:
            signals[i] = 0.25  # Long
        elif close[i] < kama[i] and rsi[i] < 50 and not trend_up:
            signals[i] = -0.25  # Short
        else:
            signals[i] = 0.0  # Flat
    
    return signals