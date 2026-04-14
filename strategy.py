#!/usr/bin/env python3
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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using previous day's OHLC)
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Calculate 6-hour EMA (21-period) for trend filter
    alpha = 2 / (21 + 1)
    ema = np.full(n, np.nan)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    
    # Calculate 6-hour RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.full(n, np.nan)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Align daily pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema[i]) or
            np.isnan(rsi[i]) or
            np.isnan(pivot_6h[i]) or
            np.isnan(r1_6h[i]) or
            np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or
            np.isnan(s2_6h[i]) or
            np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (avoid chop)
        if i >= 20:
            atr_est = np.mean(np.abs(high[i-19:i+1] - low[i-19:i+1]))
            if atr_est < 0.005 * close[i]:
                signals[i] = 0.0
                continue
        
        if position == 0:
            # Long: Price above EMA (uptrend) AND RSI > 55 (bullish momentum) AND touches S1 support
            if close[i] > ema[i] and rsi[i] > 55 and low[i] <= s1_6h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price below EMA (downtrend) AND RSI < 45 (bearish momentum) AND touches R1 resistance
            elif close[i] < ema[i] and rsi[i] < 45 and high[i] >= r1_6h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below EMA OR RSI < 40 OR touches R1 resistance (take profit)
            if close[i] < ema[i] or rsi[i] < 40 or high[i] >= r1_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above EMA OR RSI > 60 OR touches S1 support (take profit)
            if close[i] > ema[i] or rsi[i] > 60 or low[i] <= s1_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Pivot_EMA_RSI_S1S1"
timeframe = "6h"
leverage = 1.0