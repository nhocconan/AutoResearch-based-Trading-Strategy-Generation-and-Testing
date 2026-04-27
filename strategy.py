#!/usr/bin/env python3
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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day RSI (14-period)
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(df_1d['close'].values, np.nan)
    avg_loss = np.full_like(df_1d['close'].values, np.nan)
    
    if len(df_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(df_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_1d = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 100)
    
    # Calculate 1-day Moving Average (50-period) for trend
    ma_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        for i in range(49, len(df_1d)):
            ma_50_1d[i] = np.mean(df_1d['close'].values[i-49:i+1])
    
    # Align 1d indicators to 12h timeframe
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_50_1d)
    
    # Calculate 12-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 12
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(ma_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume
        vol_filter = vol_ratio > 1.8
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum) and price above MA50 with volume
            if rsi_14_1d_aligned[i] > 55 and price > ma_50_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI < 45 (bearish momentum) and price below MA50 with volume
            elif rsi_14_1d_aligned[i] < 45 and price < ma_50_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI drops below 40 or volume drops significantly
            if rsi_14_1d_aligned[i] < 40 or vol_ratio < 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI rises above 60 or volume drops significantly
            if rsi_14_1d_aligned[i] > 60 or vol_ratio < 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_RSI14_MA50_Volume"
timeframe = "12h"
leverage = 1.0