#!/usr/bin/env python3
# 12h_1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: 12h KAMA direction (trend) + RSI momentum + 1d Choppiness regime filter.
# In trending markets (CHOP < 38.2), follow KAMA direction with RSI confirmation.
# In ranging markets (CHOP > 61.8), fade extremes with RSI mean reversion.
# Uses 1d Choppiness to adapt to market regime, reducing false signals in chop.
# Target: 15-30 trades/year per symbol for low fee attrition.

name = "12h_1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "12h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h KAMA (adaptive moving average)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d Choppiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Trending market: CHOP < 38.2
            if chop_val < 38.2:
                # Long: price above KAMA and RSI > 50 (bullish momentum)
                if close[i] > kama[i] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA and RSI < 50 (bearish momentum)
                elif close[i] < kama[i] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8
            elif chop_val > 61.8:
                # Long: RSI oversold (< 30) mean reversion
                if rsi[i] < 30:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI overbought (> 70) mean reversion
                elif rsi[i] > 70:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if chop_val < 38.2:
                # In trend: exit if trend breaks or momentum fades
                if close[i] < kama[i] or rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit at RSI midpoint
                if rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        
        elif position == -1:
            # Short exit conditions
            if chop_val < 38.2:
                # In trend: exit if trend breaks or momentum fades
                if close[i] > kama[i] or rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit at RSI midpoint
                if rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals