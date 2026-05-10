# 1d_KAMA_Trend_RSI_Overbought_Oversold
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on daily timeframe to determine trend direction.
# In uptrend (price above KAMA), look for RSI oversold (<30) for long entries.
# In downtrend (price below KAMA), look for RSI overbought (>70) for short entries.
# This mean-reversion within trend strategy works in both bull and bear markets by trading pullbacks.
# Uses daily timeframe with weekly trend filter for higher reliability.
# Target: 15-25 trades/year to minimize fee drag.

name = "1d_KAMA_Trend_RSI_Overbought_Oversold"
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
    
    # Get daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on daily data
    # Parameters: ER length=10, Fast SC=2, Slow SC=30
    er_len = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    for i in range(er_len, len(close_1d)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = np.zeros_like(close_1d)
    fast_sc_val = 2 / (fast_sc + 1)
    slow_sc_val = 2 / (slow_sc + 1)
    sc = er * (fast_sc_val - slow_sc_val) + slow_sc_val
    sc = sc * sc  # Square for exponential smoothing
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[er_len] = close_1d[er_len]  # Start with close
    for i in range(er_len + 1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no additional delay needed)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI on daily data
    rsi_len = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    if len(close_1d) >= rsi_len:
        avg_gain[rsi_len-1] = np.mean(gain[:rsi_len])
        avg_loss[rsi_len-1] = np.mean(loss[:rsi_len])
        
        for i in range(rsi_len, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_len-1) + gain[i]) / rsi_len
            avg_loss[i] = (avg_loss[i-1] * (rsi_len-1) + loss[i]) / rsi_len
    
    rs = np.zeros_like(close_1d)
    rsi = np.zeros_like(close_1d)
    for i in range(rsi_len, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100  # Avoid division by zero
    
    # Align RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Weekly trend filter: use weekly EMA34 to determine higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34
    ema_len = 34
    alpha = 2 / (ema_len + 1)
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_len:
        ema_1w[ema_len-1] = np.mean(close_1w[:ema_len])
        for i in range(ema_len, len(close_1w)):
            ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    
    # Start after enough data for all indicators
    start_idx = max(er_len, rsi_len, ema_len)
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(ema_1w_aligned[i]):
            continue
            
        # Long conditions: price above KAMA (uptrend), RSI oversold, and weekly uptrend
        if (close[i] > kama_aligned[i] and 
            rsi_aligned[i] < 30 and 
            close[i] > ema_1w_aligned[i]):
            signals[i] = 0.25
            
        # Short conditions: price below KAMA (downtrend), RSI overbought, and weekly downtrend
        elif (close[i] < kama_aligned[i] and 
              rsi_aligned[i] > 70 and 
              close[i] < ema_1w_aligned[i]):
            signals[i] = -0.25
    
    return signals