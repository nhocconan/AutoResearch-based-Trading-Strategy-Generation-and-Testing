#!/usr/bin/env python3
# [24874] 1h_4h1d_trend_follow_v1
# Hypothesis: 1-hour strategy using 4h trend direction (EMA25) and 1d momentum (RSI) for signal direction, with 1h RSI pullback for entry timing.
# Long when 4h EMA25 up, 1d RSI > 50, and 1h RSI pulls back to < 40 then crosses back above 40.
# Short when 4h EMA25 down, 1d RSI < 50, and 1h RSI > 60 then crosses back below 60.
# Uses trend alignment to reduce whipsaw and capture momentum in both bull and bear markets.
# Target: 15-35 trades/year per symbol (~60-140 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_rsi(close, period):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan, dtype=float), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4-hour and 1-day data for context
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA25 for trend
    ema_25_4h = calculate_ema(df_4h['close'].values, 25)
    
    # Calculate 1d RSI(14) for momentum
    rsi_14_1d = calculate_rsi(df_1d['close'].values, 14)
    
    # Calculate 1h RSI(14) for entry timing
    rsi_14_1h = calculate_rsi(close, 14)
    
    # Align indicators to 1-hour timeframe
    ema_25_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_25_4h)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_25_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1h[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        ema_4h = ema_25_4h_aligned[i]
        rsi_1d = rsi_14_1d_aligned[i]
        rsi_1h = rsi_14_1h[i]
        
        if position == 1:  # Long
            # Exit: 4h trend turns down OR 1d momentum weakens
            if ema_4h < ema_25_4h_aligned[i-1] or rsi_1d < 45:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: 4h trend turns up OR 1d momentum strengthens
            if ema_4h > ema_25_4h_aligned[i-1] or rsi_1d > 55:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: 4h uptrend, 1d bullish momentum, 1h RSI pullback entry
            if (ema_4h > ema_25_4h_aligned[i-1] and  # 4h EMA rising
                rsi_1d > 50 and                     # 1d bullish momentum
                rsi_1h < 40 and                     # 1h RSI oversold
                i > 30 and rsi_14_1h[i-1] >= 40):   # Was above 40 previous bar (pullback complete)
                position = 1
                signals[i] = 0.20
            # Enter short: 4h downtrend, 1d bearish momentum, 1h RSI pullback entry
            elif (ema_4h < ema_25_4h_aligned[i-1] and  # 4h EMA falling
                  rsi_1d < 50 and                     # 1d bearish momentum
                  rsi_1h > 60 and                     # 1h RSI overbought
                  i > 30 and rsi_14_1h[i-1] <= 60):   # Was below 60 previous bar (pullback complete)
                position = -1
                signals[i] = -0.20
    
    return signals