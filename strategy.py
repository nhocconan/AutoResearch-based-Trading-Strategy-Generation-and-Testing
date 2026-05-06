#!/usr/bin/env python3
# 1d_1wKAMA_Trend_Filtered_By_RSI
# Strategy uses weekly KAMA trend direction for primary signal, filtered by daily RSI extremes
# and volume confirmation. Designed for 1d timeframe to capture major trend moves while
# avoiding counter-trend noise. Works in both bull and bear markets by following the
# weekly trend and using RSI for entry timing.
# Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing.

name = "1d_1wKAMA_Trend_Filtered_By_RSI"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA (using close prices)
    close_1w = df_1w['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1w, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Handle edge cases for volatility calculation
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align dimensions
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # using fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[29] = close_1w[29]  # start after 30 periods for stability
    for i in range(30, len(close_1w)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align weekly KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # average of first 13 gains (indices 1-13)
    avg_loss[13] = np.mean(loss[1:14])  # average of first 13 losses
    # Subsequent values using Wilder's smoothing
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to main timeframe (already daily, but ensure alignment)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Daily volume filter
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume above 1.5x 20-day average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any critical value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly KAMA, RSI not overbought, volume confirmation
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 70 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA, RSI not oversold, volume confirmation
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 30 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly KAMA or RSI overbought
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly KAMA or RSI oversold
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf