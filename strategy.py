#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA direction + RSI + weekly trend filter
# Long when KAMA rising (bullish trend), RSI < 30 (oversold), weekly close > weekly open (bullish week)
# Short when KAMA falling (bearish trend), RSI > 70 (overbought), weekly close < weekly open (bearish week)
# Uses KAMA for adaptive trend, RSI for mean reversion, weekly candle for higher timeframe bias
# Targets 20-50 total trades over 4 years (5-12/year) for low fee drag and high win rate

name = "1d_KAMA_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # This needs fixing - volatility should be rolling sum
    
    # Correct ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i < 1:
            er[i] = 0
        else:
            price_change = np.abs(close_1d[i] - close_1d[i-1])
            sum_abs_diff = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))) if i >= 1 else 0
            er[i] = price_change / sum_abs_diff if sum_abs_diff != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Weekly trend: bullish if weekly close > weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        kama_prev = kama_aligned[i-1] if i > 0 else kama_val
        rsi_val = rsi_aligned[i]
        weekly_bullish_val = weekly_bullish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: KAMA rising, RSI oversold, weekly bullish
            if kama_val > kama_prev and rsi_val < 30 and weekly_bullish_val:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI overbought, weekly bearish
            elif kama_val < kama_prev and rsi_val > 70 and not weekly_bullish_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or RSI overbought
            if kama_val < kama_prev or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or RSI oversold
            if kama_val > kama_prev or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals