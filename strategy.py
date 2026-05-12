#!/usr/bin/env python3
name = "12h_KAMA_RSI_ChopFilter_v3"
timeframe = "12h"
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
    
    # KAMA parameters
    kama_fast = 2
    kama_slow = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close, k=1)), axis=0) if False else np.abs(np.diff(close, k=1)).sum()  # placeholder for vectorized sum
    # Proper ER calculation using pandas for simplicity
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = close_series.diff(1).abs().rolling(10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    
    # Smoothing constants
    sc = (er * (2/(kama_fast+1) - 2/(kama_slow+1)) + 2/(kama_slow+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = pd.Series(close).diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False, min_periods=14).mean()
    ma_down = down.ewm(com=13, adjust=False, min_periods=14).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    rsi = rsi.values
    
    # Choppiness Index (14) - using high/low/close
    atr1 = pd.Series(high - low).rolling(14, min_periods=14).mean()
    atr2 = pd.Series(abs(high - np.roll(close, 1))).rolling(14, min_periods=14).mean()
    atr3 = pd.Series(abs(low - np.roll(close, 1))).rolling(14, min_periods=14).mean()
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = tr.rolling(14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Signals: KAMA direction + RSI extreme + Chop filter
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(14, n):  # wait for chop calculation
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI oversold, not choppy
            if kama_up[i] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, not choppy
            elif kama_down[i] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or RSI overbought
            if kama_down[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or RSI oversold
            if kama_up[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals