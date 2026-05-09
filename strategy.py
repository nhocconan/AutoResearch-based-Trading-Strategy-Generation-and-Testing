#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with RSI mean reversion and volume confirmation
# KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets
# RSI < 30 or > 70 provides mean reversion entries in the direction of trend
# Volume > 1.5x 20-period EMA ensures institutional participation
# Designed for fewer, high-quality trades with clear exit conditions
name = "4h_KAMA_RSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0]))
    volatility = np.abs(np.diff(df_1d['close'].values))
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.zeros_like(df_1d['close'].values)
    kama[0] = df_1d['close'].values[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].values[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend) and RSI oversold with volume spike
            if (price > kama_aligned[i] and rsi_aligned[i] < 30 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) and RSI overbought with volume spike
            elif (price < kama_aligned[i] and rsi_aligned[i] > 70 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if price < kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if price > kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals