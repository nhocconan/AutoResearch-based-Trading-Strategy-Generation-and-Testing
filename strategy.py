#!/usr/bin/env python3
"""
4h_KAMA_RSI_Chop_Filter_Strategy
Hypothesis: KAMA (adaptive trend) combined with RSI mean reversion and Choppiness Index regime filter captures trend continuations in trending markets and mean reversals in ranging markets. Uses 1d trend filter for multi-timeframe confirmation to work in both bull and bear regimes. Designed for low trade frequency (<50/year) to minimize fee drag.
"""

name = "4h_KAMA_RSI_Chop_Filter_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(vol != 0, np.abs(np.diff(close, 10)) / np.convolve(vol, np.ones(10), 'same'), 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr, axis=1) / (highest_high - lowest_low)) / np.log10(14)
    # Fix for first 14 values where sum is not yet defined
    chop_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) != 0, chop, 50)
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: KAMA uptrend + RSI oversold + chop > 61.8 (ranging) OR KAMA downtrend + RSI oversold + chop < 38.2 (trending mean reversion)
            if ((kama[i] > close[i] and rsi[i] < 30 and chop[i] > 61.8) or 
                (kama[i] < close[i] and rsi[i] < 30 and chop[i] < 38.2)):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA downtrend + RSI overbought + chop > 61.8 (ranging) OR KAMA uptrend + RSI overbought + chop < 38.2 (trending mean reversion)
            elif ((kama[i] < close[i] and rsi[i] > 70 and chop[i] > 61.8) or 
                  (kama[i] > close[i] and rsi[i] > 70 and chop[i] < 38.2)):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or KAMA crosses below price
            if rsi[i] > 70 or kama[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or KAMA crosses above price
            if rsi[i] < 30 or kama[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals