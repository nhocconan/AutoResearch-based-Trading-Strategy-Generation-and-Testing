#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI + Chop regime. KAMA adapts to market noise, reducing whipsaw in chop.
# RSI filters overbought/oversold. Chop regime ensures we only trade in trending markets (Chop < 38.2).
# Designed for ~20-30 trades/year with strict entry conditions to avoid overtrading.
# Works in bull (trend following) and bear (avoids false signals in chop).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Chop and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chop (14-period)
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr_1d])
    for i in range(len(atr_1d)):
        if i == 0:
            atr_1d[i] = tr_1d[0]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    hh_1d = np.full(len(close_1d), np.nan)
    ll_1d = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        hh_1d[i] = np.max(high_1d[i-13:i+1])
        ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    chop = np.full(len(close_1d), 50.0)
    for i in range(13, len(close_1d)):
        if hh_1d[i] - ll_1d[i] != 0:
            chop[i] = 100 * np.log10(sum(tr_1d[i-13:i+1]) / (hh_1d[i] - ll_1d[i])) / np.log10(14)
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if i < 14:
            if i == 1:
                avg_gain[i] = gain[0]
                avg_loss[i] = loss[0]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i-1]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i-1]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.full(len(close_1d), np.nan)
    rsi = np.full(len(close_1d), 50.0)
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # Calculate KAMA (10-period ER, 2/30 fast/slow)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=10))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)[:len(change)]
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    
    # KAMA
    kama = np.full(len(close_1d), np.nan)
    kama[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 10-period KAMA
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(kama_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        rsi_val = rsi_aligned[i]
        kama_val = kama_aligned[i]
        
        # Chop regime: only trade when trending (Chop < 38.2)
        trending = chop_val < 38.2
        
        if position == 0:
            # Long: price > KAMA and RSI < 70 (not overbought) in trending market
            if price > kama_val and rsi_val < 70 and trending:
                signals[i] = size
                position = 1
            # Short: price < KAMA and RSI > 30 (not oversold) in trending market
            elif price < kama_val and rsi_val > 30 and trending:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < KAMA or RSI > 70 (overbought) or chop > 61.8 (choppy)
            if price < kama_val or rsi_val > 70 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > KAMA or RSI < 30 (oversold) or chop > 61.8 (choppy)
            if price > kama_val or rsi_val < 30 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_RSI_Chop_Trend"
timeframe = "12h"
leverage = 1.0