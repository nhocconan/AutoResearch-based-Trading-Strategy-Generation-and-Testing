#!/usr/bin/env python3
name = "6h_1d_KAMA_RSI_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d KAMA (Kaufman Adaptive Moving Average) for trend
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) = abs(close - close[10]) / sum(abs(close - close[1])) over 10 periods
    change = np.abs(np.subtract(close_1d[10:], close_1d[:-10]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else np.zeros_like(close_1d)
    # Actually compute properly: ER for each point
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            price_change = np.abs(close_1d[i] - close_1d[i-10])
            sum_abs_change = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            er[i] = price_change / sum_abs_change if sum_abs_change != 0 else 0
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # 1d RSI (14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 6h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h volume filter: > 2x 24-period average (48h)
    vol_ma_6h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 2.0 * vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > KAMA and RSI > 50 (bullish momentum) with volume
            if (close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] > 50 and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA and RSI < 50 (bearish momentum) with volume
            elif (close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] < 50 and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < KAMA or RSI < 40 (losing momentum)
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > KAMA or RSI > 60 (losing momentum)
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h timeframe captures intermediate trends while avoiding noise.
# 1d KAMA provides adaptive trend filter that adjusts to market volatility.
# 1d RSI (50 center) confirms momentum direction.
# Volume filter ensures trades occur with participation.
# Works in bull (KAMA up, RSI>50) and bear (KAMA down, RSI<50) regimes.
# Target: 20-40 trades/year to minimize fee drag. Position size 0.25 limits risk.