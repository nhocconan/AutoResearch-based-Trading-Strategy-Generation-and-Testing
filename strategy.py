#!/usr/bin/env python3
name = "4h_KAMA_Trend_Plus_RSI_with_Chop_Filter"
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
    
    # 1d data for chop filter and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA on 4h for trend direction
    er = np.abs(np.diff(close, prepend=close[0]))
    er_sum = pd.Series(er).rolling(window=10, min_periods=10).sum()
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    efficiency_ratio = change / er_sum
    efficiency_ratio = np.where(er_sum == 0, 0, efficiency_ratio)
    sc = (efficiency_ratio * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama[0:9] = np.nan
    
    # RSI on 1d
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi[0:13] = np.nan
    
    # Chop on 1d: Chop = 100 * log10(sum(TR) / (ATR * n)) / log10(n)
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr1 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(sum_tr / (atr1 * 14)) / np.log10(14)
    chop = chop.values
    chop[0:13] = np.nan
    
    # Align indicators to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA from 1d? No, KAMA is on 4h price
    # Fix: KAMA calculated on 4h close, so align with prices directly
    kama_aligned = kama  # already on 4h index
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI < 30 (oversold) + Chop > 61.8 (ranging)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI > 70 (overbought) + Chop > 61.8 (ranging)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA or RSI > 70
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA or RSI < 30
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals