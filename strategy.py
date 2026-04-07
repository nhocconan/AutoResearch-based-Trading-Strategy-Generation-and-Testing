#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, identifying true trend direction.
# RSI filters for momentum strength, avoiding choppy markets.
# Chop filter ensures we only trade in trending regimes (Chop < 38.2).
# Works in bull markets via KAMA up + RSI > 50, in bear via KAMA down + RSI < 50.
# Target: 20-50 trades/year (80-200 total over 4 years) for 4h timeframe.

name = "4h_kama_rsi_chop_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 4h KAMA (ER=10, slow=2)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Start at period 10
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 4h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # 1d Chop Chopiness Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr1 = true_range(high_1d, low_1d, np.concatenate([[low_1d[0]], close_1d[:-1]]))
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max/min close over 14 periods
    max_close = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if atr1[i] > 0 and (max_close[i] - min_close[i]) > 0:
            chop[i] = 100 * np.log10(atr1[i] * 14 / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    # Pad chop to match length
    chop = np.concatenate([np.full(13, np.nan), chop])
    
    # Align 1d indicators to 4h
    kama_4h = kama  # Already 4h
    rsi_4h = rsi    # Already 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        in_trend = chop_val < 38.2  # Trending regime
        
        if position == 1:  # Long position
            # Exit: KAMA turns down or RSI < 40
            if close[i] < kama_4h[i] or rsi_4h[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: KAMA turns up or RSI > 60
            if close[i] > kama_4h[i] or rsi_4h[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if in_trend:
                # Strong uptrend: price > KAMA and RSI > 50
                if close[i] > kama_4h[i] and rsi_4h[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Strong downtrend: price < KAMA and RSI < 50
                elif close[i] < kama_4h[i] and rsi_4h[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals