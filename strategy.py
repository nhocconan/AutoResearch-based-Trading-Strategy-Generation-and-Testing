#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend filter with 1d RSI and volume confirmation for mean-reversion entries
# KAMA adapts to market efficiency, providing trend direction without lag
# RSI(14) < 30 for long, > 70 for short with volume > 1.3x average to confirm institutional interest
# Trend filter ensures trades align with higher timeframe momentum
# Works in bull/bear as KAMA follows price and RSI captures exhaustion
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for KAMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA parameters
    kama_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    if len(df_1d) < kama_len:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio and Smoothing Constant for KAMA
    price_change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'])).rolling(window=kama_len, min_periods=1).sum()
    er = np.where(volatility != 0, price_change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    kama_values = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    
    # RSI(14) on 1d
    rsi_len = 14
    if len(df_1d) < rsi_len:
        return np.zeros(n)
    
    delta = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1.3x average volume on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(kama_len, rsi_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d KAMA
        above_kama = close[i] > kama_1d_aligned[i]
        below_kama = close[i] < kama_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) + above KAMA + volume
            if (rsi_1d_aligned[i] < 30 and 
                above_kama and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: RSI > 70 (overbought) + below KAMA + volume
            elif (rsi_1d_aligned[i] > 70 and 
                  below_kama and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or price crosses below KAMA
            if rsi_1d_aligned[i] > 50 or close[i] < kama_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 or price crosses above KAMA
            if rsi_1d_aligned[i] < 50 or close[i] > kama_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_RSI_Volume_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0