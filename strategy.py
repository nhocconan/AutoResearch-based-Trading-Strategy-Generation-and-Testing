#!/usr/bin/env python3
"""
1d_KAMA_Direction_Plus_RSI_Filter
Hypothesis: Use KAMA direction on 1d timeframe for trend, with RSI on 1d for overbought/oversold conditions. Only take trades when KAMA trend and RSI extreme align. Weekly trend filter to avoid counter-trend trades. Designed for low trade frequency and high win rate in both bull and bear markets.
"""

name = "1d_KAMA_Direction_Plus_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === Daily KAMA for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i] + volatility[i-1]
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'][0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'][i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (no additional delay needed)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === Daily RSI for overbought/oversold ===
    delta = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Align RSI to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === Weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(ema50_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI oversold (<30), and above weekly EMA50
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and 
                close[i] > ema50_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI overbought (>70), and below weekly EMA50
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  close[i] < ema50_1d[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI overbought
            if (close[i] < kama_aligned[i] or rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI oversold
            if (close[i] > kama_aligned[i] or rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals