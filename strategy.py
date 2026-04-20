#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with RSI momentum and chop filter
# - Calculate KAMA on 1d to determine trend direction
# - Use 12h RSI for momentum confirmation (long when RSI > 50, short when RSI < 50)
# - Use 1d Choppiness Index to filter ranging markets (only trade when CHOP < 38.2)
# - Exit when trend reverses or momentum fades
# - Designed for low-frequency trading to minimize fee drag
# - Target: 12-25 trades per year per symbol (48-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for KAMA and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # ER = |Change| / Volatility, where Change = |close - close[10]|, Volatility = sum|close[i] - close[i-1]| for i=1..10
    change = np.abs(close_1d - np.roll(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0]))[:10])  # Simplified for first value
    # Proper volatility calculation over 10 periods
    volatility_full = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility_full[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
    # Avoid division by zero
    er = np.where(volatility_full != 0, change / volatility_full, 0)
    # Smoothing constants: fast = 2/(2+1), slow = 2/(30+1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate Choppiness Index on 1d
    # CHOP = 100 * log10(sum(TR over n) / (max(high) - min(low))) / log10(n)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        max_high[i] = np.max(high_1d[i-13:i+1])
        min_low[i] = np.min(low_1d[i-13:i+1])
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if max_high[i] != min_low[i] and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral value when undefined
    
    # Align 1d indicators to 12h timeframe
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h price and RSI data
    close = prices['close'].values
    
    # Calculate RSI on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(kama_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long entry: price above KAMA (uptrend) + RSI > 50 + choppy market filter (CHOP < 38.2)
            if price > kama_12h[i] and rsi[i] > 50 and chop_12h[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA (downtrend) + RSI < 50 + choppy market filter (CHOP < 38.2)
            elif price < kama_12h[i] and rsi[i] < 50 and chop_12h[i] < 38.2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA OR RSI < 40 (momentum fade)
            if price < kama_12h[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA OR RSI > 60 (momentum fade)
            if price > kama_12h[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0