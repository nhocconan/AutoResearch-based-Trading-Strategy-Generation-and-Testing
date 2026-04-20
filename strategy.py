#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR (14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily RSI (14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate daily KAMA (14, fast=2, slow=30)
    close_series = pd.Series(close_1d)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility > 0, pd.Series(change).rolling(window=10, min_periods=1).sum().values / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or 
            np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA with RSI > 50 and ATR filter
            if close_val > kama_val and rsi_val > 50 and atr_1d_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with RSI < 50 and ATR filter
            elif close_val < kama_val and rsi_val < 50 and atr_1d_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below KAMA
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above KAMA
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# KAMA_RSI_ATR_Filter_V1
# Uses KAMA for trend direction, RSI for momentum confirmation, and ATR for volatility filter
# Enters long when price > KAMA and RSI > 50
# Enters short when price < KAMA and RSI < 50
# Exits when price crosses KAMA in opposite direction
# Designed for 1d timeframe with ~10-30 trades/year
name = "KAMA_RSI_ATR_Filter_V1"
timeframe = "1d"
leverage = 1.0