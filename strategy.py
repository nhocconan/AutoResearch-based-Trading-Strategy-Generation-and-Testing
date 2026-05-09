#!/usr/bin/env python3
# Hypothesis: 4h KAMA trend + RSI momentum + Choppiness regime filter
# Long when KAMA trending up, RSI > 55, and Choppiness < 38.2 (trending market)
# Short when KAMA trending down, RSI < 45, and Choppiness < 38.2
# Exit when RSI crosses back to 50 or Choppiness > 61.8 (ranging market)
# Uses adaptive trend (KAMA) for direction, RSI for momentum strength, and Choppiness to avoid whipsaws in ranging markets
# Designed for low-frequency, high-conviction trades in both bull and bear markets
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "4h_KAMA_RSI_Choppiness_Trend"
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
    
    # Calculate 1d KAMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Efficiency Ratio (ER) for KAMA
    change = abs(df_1d['close'].diff(10))
    volatility = df_1d['close'].diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(df_1d))
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    kama = kama[~np.isnan(kama)]  # Remove any NaN from calculation
    if len(kama) < 1:
        return np.zeros(n)
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI (14) on 1h data for momentum
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    delta = df_1h['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Neutral when undefined
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi.values)
    
    # Calculate Choppiness Index (14) on 1d data for regime filter
    atr = np.zeros(len(df_1d))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    max_hh = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_ll = df_1d['low'].rolling(window=14, min_periods=14).min()
    
    chop = np.where(
        (max_hh - min_ll) != 0,
        100 * np.log10(atr.sum() / (max_hh - min_ll)) / np.log10(14),
        50
    )
    chop = pd.Series(chop, index=df_1d.index).fillna(50)
    
    # Align Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA up, RSI > 55, trending market (Choppiness < 38.2)
            if (kama_aligned[i] > kama_aligned[i-1] and 
                rsi_aligned[i] > 55 and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down, RSI < 45, trending market (Choppiness < 38.2)
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  rsi_aligned[i] < 45 and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI < 50 or ranging market (Choppiness > 61.8)
            if (rsi_aligned[i] < 50) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 50 or ranging market (Choppiness > 61.8)
            if (rsi_aligned[i] > 50) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals