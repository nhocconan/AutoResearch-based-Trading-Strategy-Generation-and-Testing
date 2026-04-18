#!/usr/bin/env python3
"""
4h_Momentum_Follow_Trend
Hypothesis: Combines EMA trend filter with momentum breakouts on 4h timeframe. 
Enters long when price crosses above EMA21 with RSI > 50 and volume spike, 
short when price crosses below EMA21 with RSI < 50 and volume spike. 
Uses 1h timeframe for trend confirmation (EMA50) to avoid counter-trend trades. 
Designed for 20-40 trades/year with strong trend capture in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA21 for entry signal
    ema21 = np.full(n, np.nan)
    k21 = 2 / (21 + 1)
    for i in range(21, n):
        if i == 21:
            ema21[i] = np.mean(close[i-21+1:i+1])
        else:
            ema21[i] = close[i] * k21 + ema21[i-1] * (1 - k21)
    
    # RSI(14) for momentum filter
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # 1h trend filter (EMA50) - load once before loop
    df_1h = get_htf_data(prices, '1h')
    ema50_1h = np.full(len(df_1h), np.nan)
    k50 = 2 / (50 + 1)
    for i in range(50, len(df_1h)):
        if i == 50:
            ema50_1h[i] = np.mean(df_1h['close'].values[i-50+1:i+1])
        else:
            ema50_1h[i] = df_1h['close'].values[i] * k50 + ema50_1h[i-1] * (1 - k50)
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema21[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above EMA21 with bullish momentum and volume
            if close[i] > ema21[i] and close[i-1] <= ema21[i-1] and rsi[i] > 50 and vol_spike[i] and ema50_1h_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below EMA21 with bearish momentum and volume
            elif close[i] < ema21[i] and close[i-1] >= ema21[i-1] and rsi[i] < 50 and vol_spike[i] and ema50_1h_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below EMA21 or momentum fades
            if close[i] < ema21[i] and close[i-1] >= ema21[i-1] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above EMA21 or momentum fades
            if close[i] > ema21[i] and close[i-1] <= ema21[i-1] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Momentum_Follow_Trend"
timeframe = "4h"
leverage = 1.0