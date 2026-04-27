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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day close prices
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full(len(close_1d), np.nan)
    lookback = 14
    for i in range(lookback - 1, len(close_1d)):
        highest_high = np.max(high_1d[i-lookback+1:i+1])
        lowest_low = np.min(low_1d[i-lookback+1:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate 1-day RSI (14-period) for momentum confirmation
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    # Initialize first average
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(len(close_1d), np.nan)
    rsi = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 0
    
    # Calculate 1-day EMA (21-period) for trend filter
    ema_21_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 21:
        alpha = 2 / (21 + 1)
        ema_21_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_21_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_21_1d[i-1]
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(21, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + RSI > 50 + price above EMA21
            if williams_r_aligned[i] < -80 and rsi_aligned[i] > 50 and price > ema_21_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) + RSI < 50 + price below EMA21
            elif williams_r_aligned[i] > -20 and rsi_aligned[i] < 50 and price < ema_21_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or RSI < 40
            if williams_r_aligned[i] > -20 or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or RSI > 60
            if williams_r_aligned[i] < -80 or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Williams_RSI_EMA21"
timeframe = "6h"
leverage = 1.0