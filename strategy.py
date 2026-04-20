#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_RSI4060_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily RSI(14) for regime filter ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[0:14] = np.nan  # First 14 values invalid
    
    # Align RSI to 1h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1h RSI(14) for entry signal ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:14] = np.nan  # First 14 values invalid
    
    # === Volume filter: 20-period average ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(rsi[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        rsi_1d_val = rsi_1d_aligned[i]
        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        
        if position == 0:
            # Long: RSI < 40 (oversold) in ranging market (daily RSI 40-60) with volume confirmation
            if rsi_val < 40 and 40 <= rsi_1d_val <= 60 and vol_ratio > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 60 (overbought) in ranging market (daily RSI 40-60) with volume confirmation
            elif rsi_val > 60 and 40 <= rsi_1d_val <= 60 and vol_ratio > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or overbought (>70)
            if rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or oversold (<30)
            if rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals