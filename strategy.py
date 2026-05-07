#!/usr/bin/env python3
name = "4h_1d_KAMA_RSI_Chop_Filter"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close (ER=10)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Choppiness Index (CHOP)
    atr_1d = np.zeros_like(close_1d)
    tr = np.maximum(high[1:] - low[1:], np.abs(close_1d[1:] - close_1d[:-1]))
    tr = np.maximum(tr, np.abs(high[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.inf], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    
    # Align daily indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 6)  # Wait for CHOP and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, choppy market (CHOP > 61.8)
            kama_up = kama_aligned[i] > kama_aligned[i-1]
            rsi_bullish = rsi_aligned[i] > 50
            choppy = chop_aligned[i] > 61.8
            vol_condition = volume[i] > vol_ma_6[i] * 1.5
            
            if kama_up and rsi_bullish and choppy and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, choppy market (CHOP > 61.8)
            elif not kama_up and rsi_aligned[i] < 50 and choppy and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down or RSI < 40
            if not (kama_aligned[i] > kama_aligned[i-1]) or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up or RSI > 60
            if not (kama_aligned[i] < kama_aligned[i-1]) or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA + RSI + Chop Filter
# - KAMA adapts to market noise, giving smoother trend than EMA
# - RSI > 50 for long, < 50 for short ensures momentum alignment
# - Choppiness Index > 61.8 filters for ranging markets where mean reversion works
# - Volume confirmation (1.5x average) ensures institutional participation
# - Works in both bull and bear markets by adapting to choppy conditions
# - Position size 0.25 targets ~25-40 trades/year, avoiding fee drag
# - Uses daily KAMA/RSI/CHOP for higher timeframe context
# - Exit when trend weakens (KAMA reverses) or RSI shows exhaustion