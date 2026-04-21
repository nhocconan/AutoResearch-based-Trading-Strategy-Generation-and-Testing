#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) on 1d timeframe to determine trend,
combined with RSI for entry timing and KAMA direction filter. Uses 1w EMA200 as trend
filter to avoid counter-trend trades. Trades only when KAMA direction aligns with
1w EMA200 trend. Designed for low trade frequency (target 10-25 trades/year) to
minimize fee drag while capturing major trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d and 1w HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d KAMA (10, 2, 30) ===
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0) if len(close_1d) > 10 else np.zeros_like(close_1d)
    # Handle first 10 elements
    direction = np.concatenate([np.full(10, np.nan), direction])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start KAMA at index 9
    for i in range(10, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # === 1d RSI(14) ===
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[13] = np.nanmean(gain[1:14])
    avg_loss[13] = np.nanmean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1w EMA200 trend filter ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Align 1d indicators to lower timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama_1d_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_aligned[i]
        ema_200_val = ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: Price above KAMA, RSI > 50, and above 1w EMA200 (uptrend)
            if (price_close > kama_val and 
                rsi_val > 50 and 
                price_close > ema_200_val):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 50, and below 1w EMA200 (downtrend)
            elif (price_close < kama_val and 
                  rsi_val < 50 and 
                  price_close < ema_200_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses KAMA in opposite direction
            if position == 1 and price_close < kama_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0