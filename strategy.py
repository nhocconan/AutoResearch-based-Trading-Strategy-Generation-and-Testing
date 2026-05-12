#!/usr/bin/env python3
name = "1d_WeeklyKAMA_Trend_12hRSI_Filter"
timeframe = "1d"
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
    
    # === 1w KAMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.zeros_like(close_1w)
    sc = np.zeros_like(close_1w)
    kama = np.zeros_like(close_1w)
    
    for i in range(10, len(close_1w)):
        if i >= 10:
            change_sum = np.sum(change[i-9:i+1])
            volatility_sum = np.sum(volatility[i-9:i+1])
            if volatility_sum > 0:
                er[i] = change_sum / volatility_sum
            else:
                er[i] = 0
            sc[i] = (er[i] * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
            if i == 10:
                kama[i] = close_1w[i]
            else:
                kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
        else:
            kama[i] = close_1w[i]
    
    kama_trend = kama > np.roll(kama, 1)
    kama_trend[0] = False
    
    kama_trend_aligned = align_htf_to_ltf(prices, df_1w, kama_trend.astype(float))
    
    # === 12h RSI filter for entry timing ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_12h)
    avg_loss = np.zeros_like(close_12h)
    rs = np.zeros_like(close_12h)
    rsi = np.zeros_like(close_12h)
    
    for i in range(14, len(close_12h)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_trend_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly KAMA uptrend + 12h RSI < 40 (pullback in uptrend)
            if kama_trend_aligned[i] > 0.5 and rsi_aligned[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: Weekly KAMA downtrend + 12h RSI > 60 (bounce in downtrend)
            elif kama_trend_aligned[i] < 0.5 and rsi_aligned[i] > 60:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weekly KAMA downtrend or 12h RSI > 70 (overbought)
            if kama_trend_aligned[i] < 0.5 or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weekly KAMA uptrend or 12h RSI < 30 (oversold)
            if kama_trend_aligned[i] > 0.5 or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals