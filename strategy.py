#!/usr/bin/env python3
name = "4h_KAMA_RSI_Trend_Volume_Filter"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend filter
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(df_1d['close'].diff(10).values)
    volatility = np.abs(df_1d['close'].diff(1)).rolling(10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(df_1d['close'].values, np.nan)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(kama)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for KAMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, volume spike
            if (close[i] > kama_1d_aligned[i] and 
                rsi[i] > 50 and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, volume spike
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi[i] < 50 and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down or RSI < 50
            if (close[i] < kama_1d_aligned[i] or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up or RSI > 50
            if (close[i] > kama_1d_aligned[i] or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA trend + RSI momentum + volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in choppy markets.
# RSI > 50 indicates bullish momentum, < 50 bearish momentum.
# Volume confirmation ensures strong participation in moves.
# Works in bull markets (buy when KAMA up, RSI>50, volume spike) and bear markets (sell when KAMA down, RSI<50, volume spike).
# Position size 0.25 balances risk and keeps trade frequency manageable.