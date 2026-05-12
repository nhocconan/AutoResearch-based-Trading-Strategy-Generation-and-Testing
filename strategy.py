#!/usr/bin/env python3
name = "4h_KAMA_20_RSI_14_Chop_14_Filter"
timeframe = "4h"
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
    
    # === KAMA (20) ===
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=20, min_periods=20).sum()
    er = abs(close_s.diff(10)) / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [0] * n
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # === RSI (14) ===
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # === CHOPPINESS INDEX (14) ===
    atr1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # === 1D TREND (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema34_1d_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA up + RSI > 50 + Chop < 61.8 (trending) + price above 1d EMA34
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and
                close[i] > ema34_1d_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down + RSI < 50 + Chop < 61.8 (trending) + price below 1d EMA34
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and
                  close[i] < ema34_1d_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: KAMA down OR RSI < 40 OR Chop > 61.8 (ranging) OR price below 1d EMA34
            if (close[i] < kama[i] or 
                rsi[i] < 40 or 
                chop[i] > 61.8 or
                close[i] < ema34_1d_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up OR RSI > 60 OR Chop > 61.8 (ranging) OR price above 1d EMA34
            if (close[i] > kama[i] or 
                rsi[i] > 60 or 
                chop[i] > 61.8 or
                close[i] > ema34_1d_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals