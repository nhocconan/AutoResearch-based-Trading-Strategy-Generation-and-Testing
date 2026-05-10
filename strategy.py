#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Overbought_Oversold_with_1w_Trend_Filter
# Hypothesis: Uses KAMA (adaptive moving average) on 1d timeframe to determine trend direction,
# combined with RSI(14) for entry timing (oversold in uptrend, overbought in downtrend).
# Uses 1w EMA(34) as higher timeframe trend filter to ensure alignment with weekly trend.
# Designed for low trade frequency (target: 10-25 trades/year) with strong trend persistence.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "1d_KAMA_Direction_RSI_Overbought_Oversold_with_1w_Trend_Filter"
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
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))  # |close - close[er_length]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |diff| over er_length window
    
    # Handle first er_length elements
    change_padded = np.full(n, np.nan)
    volatility_padded = np.full(n, np.nan)
    change_padded[er_length:] = change
    for i in range(er_length, n):
        volatility_padded[i] = np.sum(np.abs(np.diff(close[i-er_length:i+1])))
    
    er = np.where(volatility_padded > 0, change_padded / volatility_padded, 0)
    
    # Calculate smoothing constant SC
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # First average
    avg_gain[14] = np.mean(gain[1:15]) if n >= 15 else np.nan
    avg_loss[14] = np.mean(loss[1:15]) if n >= 15 else np.nan
    
    # Wilder smoothing
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from KAMA
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA rising (uptrend) + RSI oversold (<30) + price above 1w EMA
            if kama_rising and rsi[i] < 30 and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (downtrend) + RSI overbought (>70) + price below 1w EMA
            elif kama_falling and rsi[i] > 70 and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA falling or RSI overbought (>70)
            if kama_falling or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA rising or RSI oversold (<30)
            if kama_rising or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals