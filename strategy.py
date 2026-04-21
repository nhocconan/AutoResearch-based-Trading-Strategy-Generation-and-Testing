#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_ChopFilter_V1
Hypothesis: On daily timeframe, use KAMA for trend direction, RSI for overbought/oversold, and Choppiness Index to filter ranging markets.
Long when KAMA turns up, RSI < 30, and Choppiness > 61.8 (ranging).
Short when KAMA turns down, RSI > 70, and Choppiness > 61.8.
Exit when RSI crosses 50 or Choppiness < 38.2 (trending).
Works in bull/bear by using mean-reversion in ranging markets and avoiding trends.
Target: 10-20 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA(10) - Kaufman Adaptive Moving Average
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder
    # Correct calculation:
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14 if i >= 1 else tr[i]
    sum_atr_14 = pd.Series(atr).rolling(14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(sum_atr_14 / (max_high - min_low)) / np.log10(14), 
                    50)
    
    # Align indicators to lower timeframe (though we're using 1d as primary, we still align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long: KAMA turning up, RSI oversold, choppy market
            if (kama_aligned[i] > kama_aligned[i-1] and 
                rsi_aligned[i] < 30 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI overbought, choppy market
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  rsi_aligned[i] > 70 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses 50 or market trends
            if rsi_aligned[i] >= 50 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses 50 or market trends
            if rsi_aligned[i] <= 50 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_RSI_ChopFilter_V1"
timeframe = "1d"
leverage = 1.0