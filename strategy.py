#!/usr/bin/env python3
name = "12h_KAMA_Trend_RSI_Pullback"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (KAMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d)).cumsum() - np.abs(np.diff(close_1d, prepend=close_1d[0])).cumsum()
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))  # Correct: sum of absolute changes
    volatility = np.concatenate([[0], np.abs(np.diff(close_1d))]).cumsum()
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smooth ER
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Trend: price above/below KAMA
    trend_up = close_1d > kama
    
    # Get 12h data for entry signals (RSI pullback)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # RSI(14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_12h)
    avg_loss = np.zeros_like(close_12h)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_12h)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 12h timeframe (already 12h, but align for safety)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend + RSI pullback from overbought (RSI < 40 after >70)
            # Short: downtrend + RSI pullback from oversold (RSI > 60 after <30)
            if trend_up_aligned[i] and rsi_aligned[i] < 40:
                # Check if RSI was above 70 in last 3 bars (pullback from overbought)
                if i >= 3 and np.any(rsi_aligned[i-3:i] > 70):
                    signals[i] = 0.25
                    position = 1
            elif not trend_up_aligned[i] and rsi_aligned[i] > 60:
                # Check if RSI was below 30 in last 3 bars (pullback from oversold)
                if i >= 3 and np.any(rsi_aligned[i-3:i] < 30):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: trend changes or RSI overbought
            if not trend_up_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend changes or RSI oversold
            if trend_up_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals