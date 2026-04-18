#!/usr/bin/env python3
"""
4h RSI + Volume Spike + Trend Filter
Hypothesis: RSI extremes (oversold/overbought) combined with volume spikes indicate potential reversals or continuations with institutional interest. A trend filter (price vs EMA50) ensures we trade in the direction of the higher-timeframe trend, improving win rate in both bull and bear markets. Low trade frequency due to strict three-condition entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    for i in range(len(close)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i) + gain[i]) / (i + 1)
                avg_loss[i] = (avg_loss[i-1] * (i) + loss[i]) / (i + 1)
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            ema50_1d[i] = close_1d[i]
        else:
            ema50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema50_1d[i-1] * (49 / (50 + 1)))
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RSI on 4h
    rsi = calculate_rsi(close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: RSI oversold (<30) + price above EMA50 (uptrend) + volume spike
            if (rsi_val < 30 and 
                close[i] > ema50_val and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) + price below EMA50 (downtrend) + volume spike
            elif (rsi_val > 70 and 
                  close[i] < ema50_val and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought (>70) or price crosses below EMA50
            if rsi_val > 70 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) or price crosses above EMA50
            if rsi_val < 30 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_VolumeSpike_EMA50Trend"
timeframe = "4h"
leverage = 1.0