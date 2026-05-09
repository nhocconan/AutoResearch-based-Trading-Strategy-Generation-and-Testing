#!/usr/bin/env python3
# 1D_1W_KAMA_RSI_Trend_Signal
# Hypothesis: On 1d timeframe, use KAMA trend direction (bullish/bearish) combined with RSI extremes and weekly pivot levels for high-probability entries.
# In bullish regime (KAMA rising), buy when RSI < 30 (oversold) and price > weekly S2 (support).
# In bearish regime (KAMA falling), sell when RSI > 70 (overbought) and price < weekly R2 (resistance).
# Uses weekly pivot levels as dynamic support/resistance and RSI for mean reversion within the trend.
# Designed for 10-25 trades/year on 1d timeframe with strong edge in both bull and bear markets.

name = "1D_1W_KAMA_RSI_Trend_Signal"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # Need 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio and KAMA
    er = np.zeros_like(close_1d)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period volatility
    # Handle first 10 values
    er[:10] = 0
    for i in range(10, len(close_1d)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (already on 1d, but need to align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # First average
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else 0
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else 0
    
    # Wilder smoothing
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close_1d)
    rsi = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, S2 = P-(H-L), R2 = P+(H-L)
    pivot = (high_1w + low_1w + close_1w) / 3
    weekly_range = high_1w - low_1w
    weekly_R2 = pivot + weekly_range  # R2 = P + (H-L)
    weekly_S2 = pivot - weekly_range  # S2 = P - (H-L)
    
    # Align weekly pivot levels to 1d timeframe
    weekly_R2_aligned = align_htf_to_ltf(prices, df_1w, weekly_R2)
    weekly_S2_aligned = align_htf_to_ltf(prices, df_1w, weekly_S2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Need enough data for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(weekly_R2_aligned[i]) or np.isnan(weekly_S2_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bullish trend (price > KAMA) AND RSI oversold (<30) AND price above weekly S2 (support)
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 30 and close[i] > weekly_S2_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish trend (price < KAMA) AND RSI overbought (>70) AND price below weekly R2 (resistance)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 70 and close[i] < weekly_R2_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns bearish OR RSI overbought (>70) OR price breaks below weekly S2
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70 or close[i] < weekly_S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns bullish OR RSI oversold (<30) OR price breaks above weekly R2
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30 or close[i] > weekly_R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals