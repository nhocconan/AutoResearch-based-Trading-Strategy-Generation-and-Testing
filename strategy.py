#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI + chop filter with 1w EMA50 trend filter
# Long when KAMA rising, RSI > 50, chop < 61.8 (trending), and price > 1w EMA50
# Short when KAMA falling, RSI < 50, chop < 61.8 (trending), and price < 1w EMA50
# Exit when opposite condition or chop > 61.8 (ranging)
# Uses adaptive trend (KAMA), momentum (RSI), regime (chop), and multi-timeframe trend (1w EMA)
# Designed to capture trending moves while avoiding ranging markets
# Target: 40-80 total trades over 4 years (10-20/year) with size 0.25

name = "1d_KAMA_RSI_Chop_1wEMA50_Trend"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d KAMA (adaptive moving average)
    # Efficiency ratio: |price change| / sum of absolute price changes
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(10, len(change)):  # 10-period ER
        if np.sum(volatility[i-9:i+1]) > 0:
            er[i] = change[i] / np.sum(volatility[i-9:i+1])
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d Chopiness Index(14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((max_high - min_low) > 0, chop, 50)  # default to 50 when range is zero
    
    # Get 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI > 50, chop < 61.8 (trending), price > 1w EMA50
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI < 50, chop < 61.8 (trending), price < 1w EMA50
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 50 OR chop > 61.8 (ranging) OR price < 1w EMA50
            if (kama[i] < kama[i-1] or 
                rsi[i] < 50 or 
                chop[i] > 61.8 or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 50 OR chop > 61.8 (ranging) OR price > 1w EMA50
            if (kama[i] > kama[i-1] or 
                rsi[i] > 50 or 
                chop[i] > 61.8 or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals