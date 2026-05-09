#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_RSI_Filter_Chop_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h KAMA direction + RSI + Choppiness regime filter.
    Long when KAMA trending up, RSI > 50, and choppy market (CHOP > 61.8).
    Short when KAMA trending down, RSI < 50, and choppy market (CHOP > 61.8).
    Exit when KAMA direction changes or RSI crosses 50.
    Designed for ~80-120 total trades over 4 years to avoid fee drag.
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA calculation
    close_series = pd.Series(close)
    direction = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(10).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2
    kama = close_series.copy()
    for i in range(1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    
    # RSI calculation
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Choppiness Index calculation
    atr = pd.Series(np.maximum.reduce([
        high - low,
        np.abs(high - close_series.shift()),
        np.abs(low - close_series.shift())
    ]))
    atr_sum = atr.rolling(14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama_values[i] > kama_values[i-1]
        kama_down = kama_values[i] < kama_values[i-1]
        rsi_above_50 = rsi_values[i] > 50
        rsi_below_50 = rsi_values[i] < 50
        choppy = chop_values[i] > 61.8
        
        if position == 0:
            if kama_up and rsi_above_50 and choppy:
                signals[i] = 0.25
                position = 1
            elif kama_down and rsi_below_50 and choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            if not (kama_up and rsi_above_50 and choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if not (kama_down and rsi_below_50 and choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals