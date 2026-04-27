#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Bound_1wTrend_Filter
Hypothesis: Use daily KAMA to determine trend direction, enter only when RSI is within bounds (avoiding extremes) and weekly trend confirms. Exit when KAMA flips or RSI reaches extreme. Designed to capture trending moves while avoiding chop and overextension. Works in bull via trend continuation, in bear via avoiding false signals during downturns.
"""
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
    volume = prices['volume'].values
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA (ER=10, FA=2, SC=30) on daily close
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0) if len(close_1d) > 10 else np.array([1e-10])
    volatility = np.concatenate([np.full(10, np.nan), volatility]) if len(close_1d) > 10 else np.full(len(close_1d), np.nan)
    er = np.where(volatility[-len(direction):] != 0, direction / volatility[-len(direction):], 0)
    er = np.concatenate([np.full(10, np.nan), er])
    sc = (er * (2/10 - 1/30) + 1/30) ** 2
    kama = np.full_like(close_1d, np.nan)
    if len(close_1d) > 10:
        kama[10] = close_1d[10]
        for i in range(11, len(close_1d)):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 / (20+1)) + (ema_20_1w[i-1] * (20-1) / (20+1))
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need KAMA (10), RSI (14), weekly EMA (20)
    start_idx = max(20, 14, 10)  # 20 for weekly alignment
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought (<60), weekly uptrend (price > weekly EMA20)
            if (price > kama_aligned[i] and 
                rsi_aligned[i] < 60 and 
                price > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold (>40), weekly downtrend (price < weekly EMA20)
            elif (price < kama_aligned[i] and 
                  rsi_aligned[i] > 40 and 
                  price < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below KAMA OR RSI overbought (>=70)
            if price < kama_aligned[i] or rsi_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above KAMA OR RSI oversold (<=30)
            if price > kama_aligned[i] or rsi_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_KAMA_Trend_RSI_Bound_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0