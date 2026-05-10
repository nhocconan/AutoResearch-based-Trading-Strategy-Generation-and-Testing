#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Bounce_With_RSI
# Hypothesis: Camarilla pivot levels (R3/S3) on daily timeframe act as strong support/resistance.
# Long when price touches S3 with RSI < 30 (oversold) and closes above S3.
# Short when price touches R3 with RSI > 70 (overbought) and closes below R3.
# Uses RSI for momentum confirmation to avoid false breakouts. Designed for low trade frequency.
# Works in both bull and bear markets by fading extremes at institutional pivot levels.

name = "4h_Camarilla_Pivot_Bounce_With_RSI"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500)
    # S3 = C - ((H-L) * 1.2500), S4 = C - ((H-L) * 1.5000)
    # where C, H, L are from previous day
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Shift by 1 to use previous day's values (no look-ahead)
    prev_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    prev_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prev_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    
    # Calculate Camarilla levels for previous day
    H_L = prev_high - prev_low
    R3 = prev_close + (H_L * 1.2500)
    S3 = prev_close - (H_L * 1.2500)
    
    # Align daily levels to 4h timeframe (waits for daily bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate RSI on 4h chart (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) and Camarilla levels
    start_idx = max(14, 1)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price touches S3 and RSI oversold (<30), closes above S3
            if low[i] <= S3_aligned[i] and rsi[i] < 30 and close[i] > S3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches R3 and RSI overbought (>70), closes below R3
            elif high[i] >= R3_aligned[i] and rsi[i] > 70 and close[i] < R3_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S3 or RSI overbought (>70)
            if close[i] < S3_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R3 or RSI oversold (<30)
            if close[i] > R3_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals