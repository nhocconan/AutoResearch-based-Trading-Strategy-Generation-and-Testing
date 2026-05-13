#!/usr/bin/env python3
"""
1d_KAMA_Trend_Regime_Filter
Hypothesis: Use daily KAMA (Kaufman Adaptive Moving Average) for trend direction, 
combined with RSI(14) mean reversion and Choppiness Index regime filter. 
Long when KAMA upward, RSI < 40, and CHOP > 61.8 (range); short when KAMA downward, 
RSI > 60, and CHOP > 61.8. This captures mean reversion in ranging markets 
while avoiding trending markets where mean reversion fails. 
Designed for 1d timeframe to limit trades (<25/year) and avoid fee drag.
"""

name = "1d_KAMA_Trend_Regime_Filter"
timeframe = "1d"
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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    price_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_sum = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility_sum[i] = volatility_sum[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    er = np.where(volatility_sum > 0, price_change / volatility_sum, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index (CHOP)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - sum of TR over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(ATR14 / (HH - LL)) / log10(14)
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    chop = np.where(hl_range > 0, 100 * np.log10(atr14 / hl_range) / np.log10(14), 50)
    
    # Align CHOP to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_aligned[i-1]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # LONG: KAMA upward, RSI oversold (<40), ranging market
            if kama_up and rsi_aligned[i] < 40 and ranging:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA downward, RSI overbought (>60), ranging market
            elif kama_down and rsi_aligned[i] > 60 and ranging:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI overbought or market trends
            if not kama_up or rsi_aligned[i] > 60 or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI oversold or market trends
            if not kama_down or rsi_aligned[i] < 40 or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals